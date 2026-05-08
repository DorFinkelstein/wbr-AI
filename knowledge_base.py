"""
SQLite-backed store with two responsibilities:

  1. Persistent win/loss ledger  – remember every judge outcome so we never
     re-query the same pair twice (saves API calls and Claude tokens).

  2. Replay buffer              – store (state, action, reward, next_state,
     done) transitions for DQN training.
"""

import os
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Transition:
    state: str        # item to beat
    action: str       # our guess
    reward: float
    next_state: str   # same as action when we win; "" when done
    done: bool


class KnowledgeBase:
    def __init__(self, db_path: str, replay_buffer_size: int = 20_000):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.replay_buffer_size = replay_buffer_size
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS outcomes (
                item1       TEXT NOT NULL,
                item2       TEXT NOT NULL,
                item2wins   INTEGER NOT NULL,
                reason      TEXT,
                source      TEXT,
                ts          REAL,
                PRIMARY KEY (item1, item2)
            );

            CREATE TABLE IF NOT EXISTS replay_buffer (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                state       TEXT NOT NULL,
                action      TEXT NOT NULL,
                reward      REAL NOT NULL,
                next_state  TEXT NOT NULL,
                done        INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_outcomes_item1
                ON outcomes(item1);
        """)
        self.conn.commit()

    # ── Outcome ledger ─────────────────────────────────────────────────────────

    def record_outcome(
        self,
        item1: str,
        item2: str,
        wins: bool,
        reason: str = "",
        source: str = "unknown",
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO outcomes
               (item1, item2, item2wins, reason, source, ts)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (item1.lower(), item2.lower(), int(wins), reason, source, time.time()),
        )
        self.conn.commit()

    def lookup(self, item1: str, item2: str) -> Optional[bool]:
        """Return cached win/loss or None if unseen."""
        row = self.conn.execute(
            "SELECT item2wins FROM outcomes WHERE item1=? AND item2=?",
            (item1.lower(), item2.lower()),
        ).fetchone()
        return bool(row[0]) if row is not None else None

    def known_winners_for(self, item1: str) -> List[str]:
        """All items known to beat item1."""
        rows = self.conn.execute(
            "SELECT item2 FROM outcomes WHERE item1=? AND item2wins=1",
            (item1.lower(),),
        ).fetchall()
        return [r[0] for r in rows]

    def win_rate(self, item2: str) -> float:
        """Overall fraction of item types this item has beaten."""
        row = self.conn.execute(
            "SELECT COUNT(*), SUM(item2wins) FROM outcomes WHERE item2=?",
            (item2.lower(),),
        ).fetchone()
        n, wins = row
        if not n:
            return 0.5  # optimistic prior
        return wins / n

    # ── Replay buffer ──────────────────────────────────────────────────────────

    def push_transition(self, t: Transition) -> None:
        self.conn.execute(
            """INSERT INTO replay_buffer (state, action, reward, next_state, done)
               VALUES (?, ?, ?, ?, ?)""",
            (t.state, t.action, t.reward, t.next_state, int(t.done)),
        )
        # Keep buffer bounded
        self.conn.execute(
            """DELETE FROM replay_buffer WHERE id IN (
                   SELECT id FROM replay_buffer
                   ORDER BY id ASC
                   LIMIT MAX(0, (SELECT COUNT(*) FROM replay_buffer) - ?)
               )""",
            (self.replay_buffer_size,),
        )
        self.conn.commit()

    def sample_transitions(self, batch_size: int) -> List[Transition]:
        rows = self.conn.execute(
            """SELECT state, action, reward, next_state, done
               FROM replay_buffer
               ORDER BY RANDOM()
               LIMIT ?""",
            (batch_size,),
        ).fetchall()
        return [
            Transition(
                state=r[0],
                action=r[1],
                reward=r[2],
                next_state=r[3],
                done=bool(r[4]),
            )
            for r in rows
        ]

    def buffer_size(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM replay_buffer"
        ).fetchone()[0]

    def total_outcomes(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM outcomes"
        ).fetchone()[0]

    def close(self) -> None:
        self.conn.close()
