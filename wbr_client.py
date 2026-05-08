"""
Three interchangeable judges for WBR:

  RealWBRClient   – hits the live whatbeatsrock.com API
  GeminiJudge     – uses Gemini (free tier) to simulate the narrator
  ClaudeJudge     – uses Claude to simulate the narrator

All offline judges expose the same interface:
    reset()
    judge(item1: str, item2: str) -> JudgeResult
"""

import json
import os
import time
import uuid
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
    """Sends guesses to the live whatbeatsrock.com API."""

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
        self._gid: str = str(uuid.uuid4())

    def reset(self) -> None:
        """Call at the start of each new game to get a fresh session UUID."""
        self._gid = str(uuid.uuid4())

    def judge(self, item1: str, item2: str) -> JudgeResult:
        # Rate-limit
        wait = self.cfg.request_delay - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

        url = self.cfg.api_base_url.rstrip("/") + self.cfg.api_path
        body = {
            self.cfg.api_field_prev: item1,
            self.cfg.api_field_guess: item2,
            self.cfg.api_field_gid: self._gid,
        }

        resp = self.session.post(url, json=body, timeout=15)
        if resp.status_code == 404:
            raise RuntimeError(
                f"\n\n  404 Not Found: {url}\n"
                "  The API path in config.py is wrong.\n"
                "  Fix: open whatbeatsrock.com in Chrome → F12 → Network tab →\n"
                "  play one round → click the POST request → copy the path from\n"
                "  the Headers tab → update api_path in config.py.\n"
            )
        resp.raise_for_status()
        payload = resp.json()

        # Response is nested: {"data": {"guess_wins": ..., "reason": ...}}
        data = payload.get(self.cfg.api_response_data_key, payload)
        wins = bool(data.get(self.cfg.api_field_wins, False))
        reason = data.get("reason", "")
        return JudgeResult(item1=item1, item2=item2, item2wins=wins, reason=reason)


# ── Shared prompt ─────────────────────────────────────────────────────────────

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

_JUDGE_USER = 'Does "{item2}" beat "{item1}"?'


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Gemini judge (free tier) ───────────────────────────────────────────────────

class GeminiJudge:
    """
    Mimics the WBR narrator using Google Gemini.

    Free tier: https://aistudio.google.com/apikey
    Requires GEMINI_API_KEY env var.
    """

    def reset(self) -> None:
        pass

    def __init__(self, cfg):
        try:
            from google import generativeai as genai
        except ImportError as e:
            raise ImportError("pip install google-generativeai") from e

        self.cfg = cfg
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(
            model_name=cfg.gemini_model,
            system_instruction=_JUDGE_SYSTEM,
            generation_config={"temperature": cfg.judge_temperature, "max_output_tokens": 128},
        )
        self._cache: dict[tuple, JudgeResult] = {}

    def judge(self, item1: str, item2: str) -> JudgeResult:
        key = (item1.lower(), item2.lower())
        if key in self._cache:
            return self._cache[key]

        response = self.model.generate_content(
            _JUDGE_USER.format(item1=item1, item2=item2)
        )
        data = _parse_json_response(response.text)
        result = JudgeResult(
            item1=item1,
            item2=item2,
            item2wins=bool(data["wins"]),
            reason=data.get("reason", ""),
        )
        self._cache[key] = result
        return result


# ── Claude judge ───────────────────────────────────────────────────────────────

class ClaudeJudge:
    """
    Mimics the WBR narrator using Claude.

    Requires ANTHROPIC_API_KEY env var.
    """

    def reset(self) -> None:
        pass

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
            temperature=self.cfg.judge_temperature,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": _JUDGE_USER.format(
                item1=item1, item2=item2
            )}],
        )
        data = _parse_json_response(msg.content[0].text)
        result = JudgeResult(
            item1=item1,
            item2=item2,
            item2wins=bool(data["wins"]),
            reason=data.get("reason", ""),
        )
        self._cache[key] = result
        return result


# ── Factory ────────────────────────────────────────────────────────────────────

def make_judge(cfg):
    if not cfg.judge_backend or cfg.judge_backend == "real":
        return RealWBRClient(cfg)
    if cfg.judge_backend == "gemini":
        return GeminiJudge(cfg)
    if cfg.judge_backend == "claude":
        return ClaudeJudge(cfg)
    raise ValueError(f"Unknown judge_backend: {cfg.judge_backend!r}. Choose gemini, claude, or real.")
