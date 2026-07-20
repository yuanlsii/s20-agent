"""HTTP entry point for the S20-inspired minimal agent."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent_runtime import AgentRuntime, DemoClient, build_runtime


ROOT = Path(__file__).resolve().parent
# Cloud platforms inject PORT and require listening on every network interface.
# AGENT_* remains available for explicit local overrides.
HOST = os.getenv("AGENT_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("AGENT_PORT", "8765")))
_runtime: AgentRuntime | None = None
_runtime_error: str | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> AgentRuntime | None:
    """Build the runtime once; all requests in this process share its stores and locks."""
    global _runtime, _runtime_error
    if _runtime is not None or _runtime_error is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None or _runtime_error is not None:
            return _runtime
        try:
            _runtime = build_runtime(ROOT)
        except RuntimeError as exc:
            _runtime_error = str(exc)
    return _runtime


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class AgentHandler(BaseHTTPRequestHandler):
    server_version = "S20Agent/0.1"
    protocol_version = "HTTP/1.1"

    def _send(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send(status, json_bytes(payload), "application/json; charset=utf-8")

    def _start_event_stream(self) -> None:
        """Start a chunked SSE response so progress reaches the browser immediately."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _write_event(self, event_type: str, **payload: Any) -> None:
        """Write one JSON SSE event using HTTP/1.1 chunk framing."""
        body = json.dumps({"type": event_type, **payload}, ensure_ascii=False)
        data = f"data: {body}\n\n".encode("utf-8")
        chunk = f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n"
        self.wfile.write(chunk)
        self.wfile.flush()

    def _finish_event_stream(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _stream_chat(self, runtime: AgentRuntime, session_id: str, message: str) -> None:
        """Run the loop while forwarding safe progress and answer chunks as SSE."""
        self._start_event_stream()
        try:
            def on_event(kind: str, detail: str) -> None:
                if kind == "answer":
                    # The provider adapter is intentionally non-streaming today. Chunk
                    # the final text here so the UI still gets a progressive answer.
                    for index in range(0, len(detail), 8):
                        self._write_event("answer_delta", delta=detail[index : index + 8])
                        time.sleep(0.015)
                    return
                self._write_event("progress", message=detail)

            result = runtime.run(session_id, message, on_event=on_event)
            self._write_event("done", result=result)
        except Exception as exc:
            try:
                self._write_event("error", error=str(exc))
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            try:
                self._finish_event_stream()
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _read_json(self, allow_empty: bool = False) -> dict[str, Any]:
        """Read a small JSON request body and reject oversized or non-object payloads."""
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0 and allow_empty:
            return {}
        if length <= 0 or length > 32_000:
            raise ValueError("request body must be between 1 and 32000 bytes")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _require_runtime(self) -> AgentRuntime | None:
        """Turn missing provider configuration into a clear HTTP 503 response."""
        runtime = get_runtime()
        if runtime is None:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": _runtime_error or "agent is not configured"},
            )
        return runtime

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/api/health":
            runtime = get_runtime()
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "configured": runtime is not None,
                    "demo_mode": isinstance(runtime.model, DemoClient) if runtime else False,
                    "provider": os.getenv(
                        "SUB2API_BASE_URL", "https://sub2api-yuanlsii.zeabur.app/v1"
                    ),
                    "error": _runtime_error,
                },
            )
            return

        if path == "/api/sessions":
            # The browser uses summaries for its clickable session list.
            runtime = self._require_runtime()
            if runtime is not None:
                self._send_json(
                    HTTPStatus.OK, {"sessions": runtime.sessions.list_sessions()}
                )
            return

        if path.startswith("/api/sessions/"):
            # Only expose user/assistant text to the UI; tool internals stay server-side.
            runtime = self._require_runtime()
            if runtime is None:
                return
            session_id = path.removeprefix("/api/sessions/")
            if not runtime.sessions.exists(session_id):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
                return
            session = runtime.sessions.load(session_id)
            visible_messages = [
                {"role": message["role"], "content": message.get("content", "")}
                for message in session.messages
                if message.get("role") in {"user", "assistant"}
                and isinstance(message.get("content"), str)
                and message.get("content")
            ]
            self._send_json(
                HTTPStatus.OK,
                {
                    "session": {
                        "id": session.id,
                        "messages": visible_messages,
                        "created_at": session.created_at,
                        "updated_at": session.updated_at,
                    }
                },
            )
            return

        if path == "/" or path == "/index.html":
            html = (ROOT / "static" / "index.html").read_bytes()
            self._send(HTTPStatus.OK, html, "text/html; charset=utf-8")
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path not in {"/api/chat", "/api/chat/stream", "/api/sessions"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            runtime = self._require_runtime()
            if runtime is None:
                return

            if path == "/api/sessions":
                # Creating a session is separate from sending its first message so the
                # UI can show an empty conversation immediately.
                data = self._read_json(allow_empty=True)
                requested_id = data.get("session_id")
                session_id = str(requested_id).strip() if requested_id else None
                session = runtime.sessions.create(session_id)
                self._send_json(
                    HTTPStatus.CREATED,
                    {
                        "session": {
                            "id": session.id,
                            "preview": "",
                            "message_count": 0,
                            "created_at": session.created_at,
                            "updated_at": session.updated_at,
                        }
                    },
                )
                return

            # This is the HTTP boundary for the four-step AgentRuntime loop.
            data = self._read_json()
            message = str(data.get("message", ""))
            session_id = str(data.get("session_id") or f"session_{uuid.uuid4().hex[:12]}")
            if path == "/api/chat/stream":
                self._stream_chat(runtime, session_id, message)
                return
            result = runtime.run(session_id, message)
            self._send_json(HTTPStatus.OK, result)
        except RuntimeError as exc:
            status = HTTPStatus.CONFLICT if "busy" in str(exc) else HTTPStatus.BAD_GATEWAY
            self._send_json(status, {"error": str(exc)})
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc}"})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # Keep the HTTP process alive for one bad request.
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AgentHandler)
    print(f"S20 agent listening on http://{HOST}:{PORT}")
    print("Set DEMO_MODE=1 for deterministic local testing, or configure Sub2API first.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
