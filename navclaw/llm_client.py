"""OpenAI-compatible client with exact-request retry semantics."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class LLMError(RuntimeError):
    pass


class LLMFatalError(LLMError):
    pass


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)) or default)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)) or default)
    except ValueError:
        return default


def make_chat_endpoint(api_base: Optional[str]) -> str:
    if not api_base:
        return "https://api.openai.com/v1/chat/completions"
    base = api_base.rstrip("/")
    if base.endswith("/keys"):
        raise LLMFatalError("api_base_looks_like_key_management_page")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


class LLMClient:
    def __init__(self, log_dir: Path):
        self.api_key = _env("NAVCLAW_LLM_API_KEY") or _env("APEXNAV_VLM_API_KEY") or _env(
            "OPENAI_API_KEY"
        )
        self.api_base = _env("NAVCLAW_LLM_API_BASE") or _env("APEXNAV_VLM_API_BASE") or _env(
            "OPENAI_BASE_URL"
        )
        self.model = _env("NAVCLAW_LLM_MODEL") or _env("APEXNAV_VLM_MODEL") or "gpt-5.4"
        self.timeout = max(1.0, _float_env("NAVCLAW_LLM_TIMEOUT", 600.0))
        self.max_retries = max(0, _int_env("NAVCLAW_LLM_API_RETRIES", 8))
        self.retry_sleep = max(0.1, _float_env("NAVCLAW_LLM_RETRY_SLEEP", 2.0))
        self.retry_max_sleep = max(
            self.retry_sleep, _float_env("NAVCLAW_LLM_RETRY_MAX_SLEEP", 30.0)
        )
        self.temperature = _float_env("NAVCLAW_LLM_TEMPERATURE", 0.0)
        self.endpoint = make_chat_endpoint(self.api_base)
        self.log_path = Path(log_dir) / "api_attempts.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.api_key or self.api_base)

    def _append_log(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._log_lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def chat(
        self, messages: List[Dict[str, Any]], request_id: str
    ) -> Tuple[str, str, int]:
        if not self.configured:
            raise LLMFatalError("llm_api_not_configured")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        body = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        body_hash = hashlib.sha256(body).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _env("NAVCLAW_LLM_USER_AGENT", "OpenAI/Python 1.0.0") or "OpenAI/Python 1.0.0",
        }
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        last_error = "unknown_error"
        retryable_codes = {408, 409, 425, 429, 500, 502, 503, 504}
        for attempt in range(1, self.max_retries + 2):
            started = time.time()
            record: Dict[str, Any] = {
                "request_id": request_id,
                "request_body_sha256": body_hash,
                "attempt": attempt,
                "model": self.model,
                "started_at": started,
            }
            try:
                request = urllib.request.Request(
                    self.endpoint, data=body, headers=headers, method="POST"
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_body = response.read().decode("utf-8")
                    status = int(getattr(response, "status", 200))
                response_json = json.loads(response_body)
                raw = response_json["choices"][0]["message"].get("content") or ""
                record.update(
                    {
                        "status": "ok",
                        "http_status": status,
                        "elapsed_s": round(time.time() - started, 3),
                        "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    }
                )
                self._append_log(record)
                return raw, body_hash, attempt
            except urllib.error.HTTPError as exc:
                response_text = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = "http_{0}:{1}".format(exc.code, response_text)
                retryable = int(exc.code) in retryable_codes
                record.update(
                    {
                        "status": "error",
                        "http_status": int(exc.code),
                        "retryable": retryable,
                        "error": last_error,
                        "elapsed_s": round(time.time() - started, 3),
                    }
                )
                self._append_log(record)
                if not retryable:
                    raise LLMFatalError(last_error) from exc
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                last_error = "transport_error:{0}".format(exc)
                record.update(
                    {
                        "status": "error",
                        "retryable": True,
                        "error": last_error,
                        "elapsed_s": round(time.time() - started, 3),
                    }
                )
                self._append_log(record)
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                last_error = "invalid_api_response:{0}".format(exc)
                record.update(
                    {
                        "status": "error",
                        "retryable": True,
                        "error": last_error,
                        "elapsed_s": round(time.time() - started, 3),
                    }
                )
                self._append_log(record)

            if attempt <= self.max_retries:
                sleep_s = min(self.retry_max_sleep, self.retry_sleep * (2 ** (attempt - 1)))
                time.sleep(sleep_s)
        raise LLMError("llm_api_failed_after_retries:{0}".format(last_error))
