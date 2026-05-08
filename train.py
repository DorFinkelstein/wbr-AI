"""
Training entry-point.

Two phases:
  1. Offline pre-training  – Claude acts as the judge.  Fast, cheap, no rate
                             limits.  Teaches the agent the basic rules.
  2. Online fine-tuning    – real whatbeatsrock.com API.  Slow (rate-limited),
                             but teaches the agent the actual narrator's quirks.

Usage
─────
  # Phase 1 only (Claude judge, no internet needed for the game site):
  python train.py --phase offline

  # Phase 2 only (real API, pre-trained weights required):
  python train.py --phase online

  # Both phases back-to-back:
  python train.py --phase both

  # Watch the agent play after training:
  python train.py --play

Environment variables
─────────────────────
  ANTHROPIC_API_KEY   required for offline phase / ClaudeJudge
"""

import argparse
import os
import statistics
import time
from typing import List

from tqdm import tqdm

from agent import WBRAgent
from config import Config
from knowledge_base import KnowledgeBase
from vocabulary import load_vocabulary, save_vocabulary
from wbr_env import WhatBeatsRockEnv


def run_episode(
    env: WhatBeatsRockEnv,
    agent: WBRAgent,
    train: bool = True,
    render: bool = False,
) -> dict:
    obs, info = env.reset()
    tried_this_episode: List[int] = []
    total_reward = 0.0
    losses = []

    while True:
        action = agent.select_action(env._current_item, mask=tried_this_episode)
        tried_this_episode.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render:
            print(
                f"  {'✓' if info['wins'] else '✗'} "
                f"{info['guess']:30s} | "
                f"chain={info['chain_length']:3d} | "
                f"{info['reason'][:55]}"
            )

        if train:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        if terminated or truncated:
            break

    return {
        "chain_length": info["chain_length"],
        "total_reward": total_reward,
        "avg_loss": statistics.mean(losses) if losses else None,
        "epsilon": agent.epsilon,
    }


def train_phase(
    cfg: Config,
    agent: WBRAgent,
    env: WhatBeatsRockEnv,
    n_episodes: int,
    phase_name: str,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Phase: {phase_name}  ({n_episodes} episodes)")
    print(f"{'='*60}")

    chain_history: List[int] = []
    best_chain = 0

    pbar = tqdm(range(1, n_episodes + 1), desc=phase_name)
    for ep in pbar:
        result = run_episode(env, agent, train=True, render=False)
        chain = result["chain_length"]
        chain_history.append(chain)
        best_chain = max(best_chain, chain)

        recent = chain_history[-20:]
        pbar.set_postfix(
            chain=chain,
            best=best_chain,
            avg20=f"{statistics.mean(recent):.1f}",
            eps=f"{result['epsilon']:.3f}",
            loss=f"{result['avg_loss']:.4f}" if result["avg_loss"] else "n/a",
        )

        if ep % cfg.save_freq == 0:
            agent.save(cfg.model_path)
            save_vocabulary(env.vocab, cfg.vocab_path)
            tqdm.write(
                f"  [ep {ep:4d}] chain={chain:3d}  best={best_chain:3d}  "
                f"vocab={len(env.vocab)}  kb={env.kb.total_outcomes()} outcomes"
            )

    agent.save(cfg.model_path)
    save_vocabulary(env.vocab, cfg.vocab_path)
    print(f"\nPhase complete. Best chain: {best_chain}  |  Model: {cfg.model_path}")


def play(cfg: Config, agent: WBRAgent, env: WhatBeatsRockEnv, n: int = 3) -> None:
    print(f"\n{'='*60}")
    print("  Agent playing (greedy, no training)")
    print(f"{'='*60}")
    agent.epsilon = 0.0  # pure exploitation
    for i in range(n):
        print(f"\n── Game {i+1} ──")
        result = run_episode(env, agent, train=False, render=True)
        print(f"  Final chain: {result['chain_length']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WBR RL agent")
    parser.add_argument(
        "--phase",
        choices=["offline", "online", "both"],
        default="offline",
        help="Which training phase to run",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Watch the agent play after training",
    )
    parser.add_argument(
        "--offline-episodes",
        type=int,
        default=None,
        help="Override Config.offline_episodes",
    )
    parser.add_argument(
        "--online-episodes",
        type=int,
        default=None,
        help="Override Config.online_episodes",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.offline_episodes:
        cfg.offline_episodes = args.offline_episodes
    if args.online_episodes:
        cfg.online_episodes = args.online_episodes

    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.db_path), exist_ok=True)

    kb = KnowledgeBase(cfg.db_path, cfg.replay_buffer_size)
    vocab = load_vocabulary(cfg.vocab_path)

    # ── Offline phase uses Claude judge ───────────────────────────────────────
    if args.phase in ("offline", "both"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not set. Required for offline phase.")
            return

        cfg.use_claude_judge = True
        env = WhatBeatsRockEnv(cfg, kb, vocab=list(vocab))
        agent = WBRAgent(cfg, env.vocab, kb)

        if os.path.exists(cfg.model_path):
            print(f"Loading existing weights from {cfg.model_path}")
            agent.load(cfg.model_path)

        train_phase(cfg, agent, env, cfg.offline_episodes, "Offline (Claude judge)")
        vocab = env.vocab  # vocab may have grown

    # ── Online phase uses real WBR API ────────────────────────────────────────
    if args.phase in ("online", "both"):
        cfg.use_claude_judge = False
        cfg.epsilon_start = 0.3   # lower epsilon for fine-tuning
        env = WhatBeatsRockEnv(cfg, kb, vocab=list(vocab))
        agent = WBRAgent(cfg, env.vocab, kb)

        if os.path.exists(cfg.model_path):
            print(f"Loading existing weights from {cfg.model_path}")
            agent.load(cfg.model_path)
            agent.epsilon = cfg.epsilon_start  # reset for fine-tuning
        else:
            print("WARNING: No pre-trained model found. Running online from scratch.")

        train_phase(cfg, agent, env, cfg.online_episodes, "Online (real WBR API)")
        vocab = env.vocab

    # ── Optional demo play ────────────────────────────────────────────────────
    if args.play:
        if "agent" not in dir():  # loaded but not trained in this run
            cfg.use_claude_judge = True
            env = WhatBeatsRockEnv(cfg, kb, vocab=list(vocab))
            agent = WBRAgent(cfg, env.vocab, kb)
            if os.path.exists(cfg.model_path):
                agent.load(cfg.model_path)
        play(cfg, agent, env)

    kb.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
