"""Claude, over plain HTTPS: the standard library only.

The app may use nothing but the standard library, so this is the Messages API
called with urllib rather than the Anthropic SDK. Three rules hold here:

* The key goes to api.anthropic.com and nowhere else. The address is fixed
  below, redirects are refused (see `app.py`), and the key travels in an
  unredirected header.
* The model gets no tools. It reads the person's notes and writes JSON
  against a schema; it cannot fetch, run or read anything.
* Nothing the provider says is passed on. A failure becomes one of a few
  fixed words in `LLMUnavailable.reason`, and the caller falls back to
  deterministic templates.
"""

from __future__ import annotations

import json
import urllib.error
from typing import Callable

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2026-10-21"
MODEL = "claude-opus-5-5"
KEY_NAME = "ANTHROPIC_API_KEY"


class LLMUnavailable(Exception):
    """Claude could not be used for this call. `reason` is a fixed word."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def ask_json(fetch: Callable[..., bytes], key: str, system: str, prompt: str,
             schema: dict, max_tokens: int, timeout: float) -> dict:
    """One Messages API call whose answer is JSON matching `schema`.

    Structured outputs (`output_config.format`) make the reply valid JSON;
    effort `low` keeps it fast enough for a 20-second call. Raises
    LLMUnavailable on every failure, never with the provider's own words.
    """
    if timeout < 2.0:
        raise LLMUnavailable("no_time")
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": schema},
        },
    }
    try:
        raw = fetch(
            API_URL,
            timeout=timeout,
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json", "anthropic-version": API_VERSION},
            key_header=("x-api-key", key),
        )
    except urllib.error.HTTPError as exc:            # first: it IS a URLError
        if exc.code in (401, 403):
            raise LLMUnavailable("key_rejected") from None
        if exc.code == 429:
            raise LLMUnavailable("rate_limited") from None
        if exc.code in (500, 502, 503, 504, 529):
            raise LLMUnavailable("overloaded") from None
        raise LLMUnavailable(f"http_{exc.code}") from None
    except TimeoutError:                             # not a URLError
        raise LLMUnavailable("timeout") from None
    except (urllib.error.URLError, OSError):
        raise LLMUnavailable("unreachable") from None

    try:
        reply = json.loads(raw)
    except ValueError:
        raise LLMUnavailable("bad_reply") from None
    stop = reply.get("stop_reason")
    if stop == "refusal":
        raise LLMUnavailable("refusal")
    if stop == "max_tokens":
        raise LLMUnavailable("truncated")
    text = next((b.get("text", "") for b in reply.get("content", []) if b.get("type") == "text"), "")
    try:
        data = json.loads(text)
    except ValueError:
        raise LLMUnavailable("bad_json") from None
    if not isinstance(data, dict):
        raise LLMUnavailable("bad_json")
    return data


#: Why Claude was not used, in words a person can read. Never the provider's.
REASONS = {
    "no_key": "No Claude key is set, so the imagery comes from built-in templates.",
    "no_time": "There was not enough time left in this call for Claude, so the imagery comes from templates.",
    "key_rejected": "The Claude key was not accepted, so the imagery comes from templates.",
    "rate_limited": "Claude is busy just now, so the imagery comes from templates.",
    "overloaded": "Claude is overloaded just now, so the imagery comes from templates.",
    "timeout": "Claude did not answer in time, so the imagery comes from templates.",
    "unreachable": "Claude could not be reached, so the imagery comes from templates.",
    "refusal": "Claude declined this material, so the imagery comes from templates.",
    "truncated": "Claude's answer was cut short, so the imagery comes from templates.",
    "bad_reply": "Claude's answer could not be read, so the imagery comes from templates.",
    "bad_json": "Claude's answer could not be read, so the imagery comes from templates.",
}


def reason_text(reason: str) -> str:
    return REASONS.get(reason, "Claude could not be used, so the imagery comes from templates.")
