"""
Gymnasium environment wrapping the WBR game.

Observation : index into the vocabulary  (the current item to beat)
Action      : index into the vocabulary  (our proposed item)
Reward      : +1 for each successful beat; 0 and episode ends on failure
Episode     : starts at "rock", ends when a guess loses

The environment checks the knowledge base first before calling the judge,
so cached pairs never cost an API call.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import Config
from knowledge_base import KnowledgeBase, Transition
from vocabulary import load_vocabulary, add_item
from wbr_client import make_judge


class WhatBeatsRockEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        cfg: Config,
        kb: KnowledgeBase,
        vocab: Optional[List[str]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.kb = kb
        self.render_mode = render_mode
        self.judge = make_judge(cfg)

        self.vocab: List[str] = vocab if vocab is not None else load_vocabulary(cfg.vocab_path)

        n = len(self.vocab)
        self.observation_space = spaces.Discrete(n)
        self.action_space = spaces.Discrete(n)

        self._current_item: str = "rock"
        self._current_idx: int = self._ensure_idx("rock")
        self._chain_length: int = 0
        self._done: bool = False

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[int, Dict]:
        super().reset(seed=seed)
        self.judge.reset()  # new gid for real client; noop for Claude
        self._current_item = "rock"
        self._current_idx = self._ensure_idx("rock")
        self._chain_length = 0
        self._done = False
        return self._current_idx, {"chain_length": 0, "current_item": "rock"}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        assert not self._done, "Call reset() before stepping after episode end."

        guess = self.vocab[action]
        item1 = self._current_item

        # Cache hit – no API call needed
        cached = self.kb.lookup(item1, guess)
        if cached is not None:
            wins = cached
            reason = "(cached)"
        else:
            result = self.judge.judge(item1, guess)
            wins = result.item2wins
            reason = result.reason
            source = self.cfg.judge_backend
            self.kb.record_outcome(item1, guess, wins, reason, source)

            if wins and self.cfg.auto_expand_vocab and guess not in self.vocab:
                self.vocab = add_item(self.vocab, guess, self.cfg.vocab_path)
                # Extend action/obs spaces to match new vocab length
                n = len(self.vocab)
                self.observation_space = spaces.Discrete(n)
                self.action_space = spaces.Discrete(n)

        reward = 1.0 if wins else 0.0
        self._done = not wins

        if wins:
            self._chain_length += 1
            prev_item = self._current_item
            self._current_item = guess
            self._current_idx = self._ensure_idx(guess)
            self.kb.push_transition(
                Transition(
                    state=prev_item,
                    action=guess,
                    reward=reward,
                    next_state=guess,
                    done=False,
                )
            )
        else:
            self.kb.push_transition(
                Transition(
                    state=item1,
                    action=guess,
                    reward=0.0,
                    next_state="",
                    done=True,
                )
            )

        if self._chain_length >= self.cfg.max_chain_length:
            self._done = True

        info = {
            "current_item": self._current_item,
            "guess": guess,
            "wins": wins,
            "reason": reason,
            "chain_length": self._chain_length,
        }

        if self.render_mode == "human":
            self._render_human(info)

        return self._current_idx, reward, self._done, False, info

    def render(self) -> None:
        pass  # rendering is inline in step() when render_mode="human"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ensure_idx(self, item: str) -> int:
        item = item.lower()
        if item not in self.vocab:
            self.vocab.append(item)
            n = len(self.vocab)
            self.observation_space = spaces.Discrete(n)
            self.action_space = spaces.Discrete(n)
        return self.vocab.index(item)

    def _render_human(self, info: Dict) -> None:
        icon = "✓" if info["wins"] else "✗"
        print(
            f"  [{icon}] {info['current_item'] if not info['wins'] else self.vocab[self._ensure_idx(info['guess'])]} "
            f"| chain={info['chain_length']} | {info['reason'][:60]}"
        )
