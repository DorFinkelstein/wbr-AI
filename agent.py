"""
DQN agent for WBR.

Architecture
────────────
  Embedding layer  :  sentence-transformers (all-MiniLM-L6-v2, frozen)
  Q-network        :  MLP([state_embed ‖ action_embed]) → scalar Q-value
  Target network   :  periodically hard-copied from online network (DQN trick)

Action selection
────────────────
  At each step we score every vocab item as a candidate answer, then pick:
    • explore  – random choice (ε-greedy)
    • exploit  – highest Q(state, candidate)
  We also give a warm-start bonus to items the knowledge-base already knows
  are winners for similar items (nearest-neighbour transfer).
"""

import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from knowledge_base import KnowledgeBase, Transition


# ── Embedding helper ──────────────────────────────────────────────────────────

class EmbeddingCache:
    """Wraps sentence-transformers and caches results in-memory."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vecs = self.model.encode(missing, show_progress_bar=False)
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ── Q-Network ─────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Agent ─────────────────────────────────────────────────────────────────────

class WBRAgent:
    def __init__(self, cfg: Config, vocab: List[str], kb: KnowledgeBase):
        self.cfg = cfg
        self.vocab = vocab
        self.kb = kb

        self.embedder = EmbeddingCache(cfg.embedding_model)
        input_dim = cfg.embedding_dim * 2  # [state ‖ action]

        self.online_net = QNetwork(input_dim, cfg.hidden_dim)
        self.target_net = QNetwork(input_dim, cfg.hidden_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.epsilon = cfg.epsilon_start
        self.steps = 0

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, current_item: str, mask: Optional[List[int]] = None) -> int:
        """
        Returns a vocab index.

        mask : optional list of vocab indices to exclude (e.g. items we just
               tried and lost with in this episode).
        """
        if random.random() < self.epsilon:
            return self._random_action(current_item, mask)
        return self._greedy_action(current_item, mask)

    def _random_action(self, current_item: str, mask: Optional[List[int]]) -> int:
        candidates = self._candidate_indices(current_item, mask)
        # Bias random exploration toward items with good overall win rates
        weights = np.array([self.kb.win_rate(self.vocab[i]) for i in candidates])
        weights = weights / weights.sum() if weights.sum() > 0 else None
        return int(np.random.choice(candidates, p=weights))

    def _greedy_action(self, current_item: str, mask: Optional[List[int]]) -> int:
        candidates = self._candidate_indices(current_item, mask)

        state_emb = self.embedder.encode_one(current_item)
        cand_texts = [self.vocab[i] for i in candidates]
        cand_embs = self.embedder.encode(cand_texts)

        state_tile = np.tile(state_emb, (len(candidates), 1))
        x = np.concatenate([state_tile, cand_embs], axis=1)
        x_t = torch.tensor(x, dtype=torch.float32)

        self.online_net.eval()
        with torch.no_grad():
            q_vals = self.online_net(x_t).numpy()
        self.online_net.train()

        # Add knowledge-base bonus: known winners get a head-start
        for k, idx in enumerate(candidates):
            item = self.vocab[idx]
            cached = self.kb.lookup(current_item, item)
            if cached is True:
                q_vals[k] += 10.0  # definitely pick known winners
            elif cached is False:
                q_vals[k] -= 10.0  # definitely avoid known losers

        best_k = int(np.argmax(q_vals))
        return candidates[best_k]

    def _candidate_indices(
        self, current_item: str, mask: Optional[List[int]]
    ) -> List[int]:
        excluded = set(mask or [])
        # Also exclude the current item itself
        try:
            excluded.add(self.vocab.index(current_item.lower()))
        except ValueError:
            pass
        candidates = [i for i in range(len(self.vocab)) if i not in excluded]
        # If vocab is large, pre-filter by embedding similarity to save time
        if len(candidates) > self.cfg.top_k_candidates * 3:
            candidates = self._embedding_shortlist(current_item, candidates)
        return candidates

    def _embedding_shortlist(self, current_item: str, indices: List[int]) -> List[int]:
        """Return indices of items most semantically related to current_item."""
        state_emb = self.embedder.encode_one(current_item)
        texts = [self.vocab[i] for i in indices]
        embs = self.embedder.encode(texts)
        # cosine similarity
        norm_s = state_emb / (np.linalg.norm(state_emb) + 1e-9)
        norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        sims = norms @ norm_s
        # Keep bottom + top: avoid very similar (likely same) and keep diverse
        top_k = self.cfg.top_k_candidates
        top_indices = np.argsort(sims)[-top_k:].tolist()
        # Also always include known winners
        known = set()
        for i, idx in enumerate(indices):
            if self.kb.lookup(current_item, self.vocab[idx]) is True:
                known.add(i)
        merged = list(set(top_indices) | known)
        return [indices[i] for i in merged]

    # ── Training step ──────────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        if self.kb.buffer_size() < self.cfg.batch_size:
            return None

        batch: List[Transition] = self.kb.sample_transitions(self.cfg.batch_size)

        states = [t.state for t in batch]
        actions = [t.action for t in batch]
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = [t.next_state if not t.done else "rock" for t in batch]
        dones = np.array([float(t.done) for t in batch], dtype=np.float32)

        state_embs = self.embedder.encode(states)
        action_embs = self.embedder.encode(actions)
        next_state_embs = self.embedder.encode(next_states)

        x = np.concatenate([state_embs, action_embs], axis=1)
        x_t = torch.tensor(x, dtype=torch.float32)

        # Target Q: use best action in next state according to target network
        # We approximate best_next_action as the current action embedding
        # (since we don't enumerate all candidates during training for speed)
        x_next = np.concatenate([next_state_embs, action_embs], axis=1)
        x_next_t = torch.tensor(x_next, dtype=torch.float32)

        with torch.no_grad():
            next_q = self.target_net(x_next_t).numpy()

        targets = rewards + self.cfg.gamma * next_q * (1.0 - dones)
        targets_t = torch.tensor(targets, dtype=torch.float32)

        q_vals = self.online_net(x_t)
        loss = self.loss_fn(q_vals, targets_t)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Decay epsilon
        self.epsilon = max(
            self.cfg.epsilon_end,
            self.epsilon * self.cfg.epsilon_decay,
        )

        return float(loss.item())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.cfg.epsilon_end)
        self.steps = ckpt.get("steps", 0)
