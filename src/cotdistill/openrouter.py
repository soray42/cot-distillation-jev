"""OpenRouter chat client with the same `chat` interface as `deepseek.DeepSeek`.

Responses are normalized to the DeepSeek shape used by the pipeline: the reasoning text is
exposed as message["reasoning_content"], and logprobs["reasoning_content"] is [] because
OpenRouter providers return logprobs for answer tokens only (checked 2026-09-29).
Pin a provider with `provider=` so all calls hit the same weights/quantization.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    pass


class OpenRouter:
    def __init__(self, api_key: str | None = None, model: str = "qwen/qwen3.5-9b", provider: str | None = None,
                 timeout: float = 600.0, max_retries: int = 4, log_path: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError("set OPENROUTER_API_KEY")
        self.model, self.provider = model, provider
        self.timeout, self.max_retries, self.log_path = timeout, max_retries, log_path
        self._lock = threading.Lock()
        self.spent_usd = 0.0
        self.n_calls = 0

    def chat(self, messages: list[dict], *, thinking: bool = True, logprobs: bool = True,
             top_logprobs: int = 20, max_tokens: int = 8000, effort: str | None = None,
             json_mode: bool = False, model: str | None = None, temperature: float | None = None,
             tag: str = "") -> dict:
        model = model or self.model
        body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens, "usage": {"include": True},
                      "reasoning": ({"effort": effort} if effort else {"enabled": True}) if thinking else {"enabled": False}}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        if temperature is not None:
            body["temperature"] = temperature
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False, "require_parameters": True}
        data = json.dumps(body).encode()
        err = ""
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(API_URL, data=data, headers={
                "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                if "choices" not in resp:
                    raise OpenRouterError(str(resp)[:300])
                self._normalize(resp)
                self._account(resp, time.time() - t0, tag)
                return resp
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
                if e.code in (400, 401, 402, 422):
                    break
            except Exception as e:
                err = repr(e)
            time.sleep(min(60, 2 ** attempt))
        raise OpenRouterError(f"{tag}: {err}")

    @staticmethod
    def _normalize(resp: dict) -> None:
        for ch in resp.get("choices", []):
            msg = ch.get("message") or {}
            msg.setdefault("reasoning_content", msg.get("reasoning") or "")
            lp = ch.get("logprobs") or {}
            lp.setdefault("reasoning_content", [])
            ch["logprobs"] = lp
        resp.setdefault("system_fingerprint", resp.get("provider"))

    def _account(self, resp: dict, secs: float, tag: str) -> None:
        usage = resp.get("usage", {})
        c = float(usage.get("cost") or 0.0)
        with self._lock:
            self.spent_usd += c
            self.n_calls += 1
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({"t": time.time(), "tag": tag, "model": resp.get("model"),
                                        "fingerprint": resp.get("provider"), "secs": round(secs, 2),
                                        "usage": usage, "usd": c}) + "\n")
