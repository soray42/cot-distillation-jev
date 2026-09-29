"""Minimal DeepSeek chat client (stdlib only) with logprobs, retries and cost accounting.

Facts checked on 2026-09-29 (api-docs.deepseek.com):
- models: deepseek-flash (V4.1), deepseek-v4-pro; thinking mode is on by default,
  disabled with {"thinking": {"type": "disabled"}}; effort via "reasoning_effort".
- top_logprobs <= 20; tokens outside the top-20 come back as -9999 (treat as censored).
- thinking mode returns logprobs for both reasoning and answer tokens.
- in multi-turn chats without tools, earlier reasoning_content is ignored, so a CoT that
  should condition a later call must be passed as plain text.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"

# USD per 1M tokens, off-peak list prices read 2026-09-29 (peak hours cost 2x).
PRICES = {
    "deepseek-flash": {"hit": 0.003, "miss": 0.15, "out": 0.60},
    "deepseek-v4-pro": {"hit": 0.022, "miss": 0.66, "out": 1.98},
}


class DeepSeekError(RuntimeError):
    pass


class DeepSeek:
    def __init__(self, api_key: str | None = None, model: str = "deepseek-flash",
                 timeout: float = 600.0, max_retries: int = 4, log_path: str | None = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise DeepSeekError("set DEEPSEEK_API_KEY")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.log_path = log_path
        self._lock = threading.Lock()
        self.spent_usd = 0.0
        self.n_calls = 0

    @staticmethod
    def cost(model: str, usage: dict) -> float:
        p = PRICES.get(model, PRICES["deepseek-flash"])
        hit = usage.get("prompt_cache_hit_tokens", 0)
        miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
        return (hit * p["hit"] + miss * p["miss"] + usage.get("completion_tokens", 0) * p["out"]) / 1e6

    def chat(self, messages: list[dict], *, thinking: bool = True, logprobs: bool = True,
             top_logprobs: int = 20, max_tokens: int = 8000, effort: str | None = None,
             json_mode: bool = False, model: str | None = None, tag: str = "") -> dict:
        model = model or self.model
        body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens, "top_p": 1.0}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        if not thinking:
            body["thinking"] = {"type": "disabled"}
        if effort:
            body["reasoning_effort"] = effort
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body).encode()
        err = ""
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(API_URL, data=data, headers={
                "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                self._account(model, resp, time.time() - t0, tag)
                return resp
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
                if e.code in (400, 401, 402, 422):
                    break
            except Exception as e:  # network / timeout
                err = repr(e)
            time.sleep(min(60, 2 ** attempt))
        raise DeepSeekError(f"{tag}: {err}")

    def _account(self, model: str, resp: dict, secs: float, tag: str) -> None:
        usage = resp.get("usage", {})
        c = self.cost(model, usage)
        with self._lock:
            self.spent_usd += c
            self.n_calls += 1
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({"t": time.time(), "tag": tag, "model": resp.get("model"),
                                        "fingerprint": resp.get("system_fingerprint"), "secs": round(secs, 2),
                                        "usage": usage, "usd": c}) + "\n")
