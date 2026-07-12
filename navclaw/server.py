"""Local persistent HTTP service for the NavClaw brain."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from .brain import DecisionError, NavClawBrain
from .llm_client import LLMError


LOGGER = logging.getLogger("navclaw.server")


class NavClawHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address, handler, brain: NavClawBrain):
        super().__init__(address, handler)
        self.brain = brain
        self.started_at = time.time()


class Handler(BaseHTTPRequestHandler):
    server: NavClawHTTPServer

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(404, {"ok": False, "error": "not_found"})
            return
        self._write_json(
            200,
            {
                "ok": True,
                "service": "navclaw_brain",
                "model": self.server.brain.llm.model,
                "llm_configured": self.server.brain.llm.configured,
                "strict_candidate_constraint": True,
                "rule_fallback_enabled": False,
                "session_count": self.server.brain.memory.session_count(),
                "uptime_s": round(time.time() - self.server.started_at, 3),
            },
        )

    def do_POST(self) -> None:
        if self.path != "/decide":
            self._write_json(404, {"ok": False, "error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._write_json(400, {"ok": False, "error": "invalid_content_length"})
            return
        if length <= 0 or length > 16 * 1024 * 1024:
            self._write_json(413, {"ok": False, "error": "request_size_out_of_range"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload_must_be_object")
            result = self.server.brain.decide(payload)
            self._write_json(200, {"ok": True, "result": result})
        except DecisionError as exc:
            LOGGER.error("decision failed: %s", exc)
            self._write_json(422, {"ok": False, "error": str(exc), "fallback": False})
        except LLMError as exc:
            LOGGER.error("LLM failed: %s", exc)
            self._write_json(503, {"ok": False, "error": str(exc), "fallback": False})
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._write_json(400, {"ok": False, "error": str(exc), "fallback": False})
        except Exception as exc:  # keep failure explicit; never synthesize an action
            LOGGER.exception("unhandled request failure")
            self._write_json(500, {"ok": False, "error": repr(exc), "fallback": False})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("NAVCLAW_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("NAVCLAW_PORT", "8765")))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(os.getenv("NAVCLAW_RUN_DIR", "runs/server")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(args.run_dir / "brain.log")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )
    brain = NavClawBrain(args.project_root, args.run_dir)
    server = NavClawHTTPServer((args.host, args.port), Handler, brain)
    LOGGER.info(
        "starting NavClaw brain host=%s port=%s run_dir=%s model=%s strict_no_fallback=true",
        args.host,
        args.port,
        args.run_dir,
        brain.llm.model,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        LOGGER.info("shutdown requested")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
