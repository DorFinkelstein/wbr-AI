"""
Two interchangeable judges for WBR:

  RealWBRClient   – hits the live whatbeatsrock.com API
  ClaudeJudge     – uses Claude to simulate the WBR narrator (for offline
                    pre-training or when the real site is unavailable)

Both expose the same interface:
    judge(item1: str, item2: str) -> JudgeResult
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class JudgeResult:
    item1: str
    item2: str
    item2wins: bool
    reason: str


# ── Real game client ───────────────────────────────────────────────────────────

class RealWBRClient:
    """
    Sends guesses to the live whatbeatsrock.com API.

    Before using this you MUST confirm the endpoint + body format by:
      1. Open https://whatbeatsrock.com in Chrome/Firefox
      2. DevTools → Network tab → play one round
      3. Find the POST request, copy URL path and JSON body keys
      4. Update Config.api_path / api_field_* accordingly
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Content-Type": "application/json",
            "Referer": cfg.api_base_url,
        })
        self._last_call = 0.0

    def judge(self, item1: str, item2: str) -> JudgeResult:
        # Rate-limit
        wait = self.cfg.request_delay - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

        url = self.cfg.api_base_url.rstrip("/") + self.cfg.api_path
        body = {
            self.cfg.api_field_item1: item1,
            self.cfg.api_field_item2: item2,
        }

        resp = self.session.post(url, json=body, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        wins = bool(data.get(self.cfg.api_field_wins, False))
        reason = data.get("reason", data.get("message", ""))
        return JudgeResult(item1=item1, item2=item2, item2wins=wins, reason=reason)


# ── Claude judge (offline / pre-training) ─────────────────────────────────────

_JUDGE_SYSTEM = """\
You are the narrator and judge for the game "What Beats Rock?".

Rules:
- Players suggest things that beat the current item in any creative or logical way.
- You decide whether the suggestion beats the current item. Be like the real WBR
  narrator: accept creative lateral thinking, physical dominance, conceptual
  superiority, or humorous logic. Reject things that are obviously weaker or have
  no plausible connection.
- Respond ONLY with a valid JSON object, no prose outside it:
  {"wins": true_or_false, "reason": "one short sentence"}
"""

_JUDGE_USER = "Does \"{item2}\" beat \"{item1}\"?"


class ClaudeJudge:
    """Mimics the WBR narrator using Claude.  Requires ANTHROPIC_API_KEY."""

    def __init__(self, cfg):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic") from e

        self.cfg = cfg
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._cache: dict[tuple, JudgeResult] = {}

    def judge(self, item1: str, item2: str) -> JudgeResult:
        key = (item1.lower(), item2.lower())
        if key in self._cache:
            return self._cache[key]

        msg = self.client.messages.create(
            model=self.cfg.claude_model,
            max_tokens=128,
            temperature=self.cfg.claude_judge_temperature,
            system=_JUDGE_SYSTEM,
            messages=[
                {"role": "user", "content": _JUDGE_USER.format(
                    item1=item1, item2=item2
                )}
            ],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        result = JudgeResult(
            item1=item1,
            item2=item2,
            item2wins=bool(data["wins"]),
            reason=data.get("reason", ""),
        )
        self._cache[key] = result
        return result


def make_judge(cfg) -> "RealWBRClient | ClaudeJudge":
    if cfg.use_claude_judge:
        return ClaudeJudge(cfg)
    return RealWBRClient(cfg)
