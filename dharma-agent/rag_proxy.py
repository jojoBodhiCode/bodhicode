"""
RAG Reverse Proxy for llama.cpp Web UI

Sits between the browser and llama-server, intercepting chat completion
requests to inject RAG context from the Buddhist knowledge base and the
Dharma Scholar system prompt. All other requests (static UI, /health,
/props, /v1/models) pass through untouched.

Usage:
    python rag_proxy.py                         # defaults: proxy on 8081, llama-server on 8080
    python rag_proxy.py --port 9090             # custom proxy port
    python rag_proxy.py --backend http://192.168.1.100:8080  # remote llama-server

Then open http://localhost:8081 in your browser for the RAG-augmented llama.cpp UI.
"""

import argparse
import json
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, urljoin
from io import BytesIO

try:
    import requests as http_requests
except ImportError:
    print("Missing 'requests'. Install with: pip install requests")
    sys.exit(1)

from prompts import SYSTEM_PROMPT

# ─── Configuration ────────────────────────────────────────────────────────────

BACKEND_URL = "http://127.0.0.1:8080"

# ─── RAG setup (lazy-loaded) ─────────────────────────────────────────────────

_rag_instance = None
_rag_lock = threading.Lock()


def get_rag():
    """Lazy-load the RAG module. Returns None if unavailable."""
    global _rag_instance
    with _rag_lock:
        if _rag_instance is None:
            try:
                from rag import DharmaRAG
                _rag_instance = DharmaRAG()
                count = _rag_instance.collection.count()
                if count > 0:
                    print(f"  [RAG] Knowledge base loaded: {count} chunks")
                else:
                    print("  [RAG] Knowledge base is empty -- responses will not be grounded.")
            except Exception as e:
                print(f"  [RAG] Could not load knowledge base: {e}")
                return None
    return _rag_instance


def augment_messages(messages):
    """
    Intercept a messages list, inject system prompt and RAG context.

    Returns (augmented_messages, sources) where sources is a list of
    metadata dicts from the retrieved chunks (for logging).
    """
    rag = get_rag()
    sources = []

    # 1. Ensure system prompt is present / replace default
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        augmented = []
        replaced = False
        for m in messages:
            if m.get("role") == "system" and not replaced:
                augmented.append({"role": "system", "content": SYSTEM_PROMPT})
                replaced = True
            else:
                augmented.append(m)
        messages = augmented

    # 2. Find the last user message and do RAG retrieval
    if rag is not None and rag.collection.count() > 0:
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            user_text = messages[last_user_idx]["content"]
            context, sources = rag.retrieve(user_text, k=5)

            if context:
                augmented_content = (
                    "The following canonical Buddhist texts are relevant to this topic. "
                    "Ground your response in these sources and cite them where appropriate.\n\n"
                    f"{context}\n\n---\n\n{user_text}"
                )
                messages = list(messages)
                messages[last_user_idx] = {
                    "role": "user",
                    "content": augmented_content,
                }

    return messages, sources


def log_sources(sources):
    """Log retrieved sources to console."""
    if sources:
        source_ids = []
        for s in sources:
            sid = s.get("text_id", s.get("source", "?"))
            if sid not in source_ids:
                source_ids.append(sid)
        print(f"  [RAG] Sources: {', '.join(source_ids)}")
    else:
        print("  [RAG] No sources retrieved for this query.")


# ─── HTTP Proxy Handler ──────────────────────────────────────────────────────

class RAGProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that proxies to llama-server with RAG injection."""

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"  {self.client_address[0]} - {format % args}")

    def _forward_request(self, method):
        """Forward a request to the backend, with RAG injection for chat completions."""
        path = self.path
        target_url = f"{BACKEND_URL}{path}"

        # Read request body if present
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Check if this is a chat completions request
        is_chat = (path == "/v1/chat/completions" and method == "POST")

        if is_chat:
            self._handle_chat_completions(body)
        else:
            self._handle_passthrough(method, target_url, body)

    def _handle_chat_completions(self, raw_body):
        """Handle chat completions with RAG injection and proper SSE streaming."""
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        messages = body.get("messages", [])
        augmented_messages, sources = augment_messages(messages)
        body["messages"] = augmented_messages
        log_sources(sources)

        is_streaming = body.get("stream", False)

        try:
            resp = http_requests.post(
                f"{BACKEND_URL}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
                stream=is_streaming,
                timeout=600,
            )
        except Exception as e:
            print(f"  [ERROR] Backend request failed: {e}")
            self.send_error(502, f"Backend error: {e}")
            return

        if is_streaming:
            # Send response headers immediately
            self.send_response(resp.status_code)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            # Stream SSE chunks directly to the socket, flushing each one
            try:
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client disconnected
            finally:
                resp.close()
        else:
            # Non-streaming: send full response
            self.send_response(resp.status_code)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            content = resp.content
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    def _handle_passthrough(self, method, target_url, body):
        """Forward a non-chat request to the backend unchanged."""
        # Build headers, skip hop-by-hop
        skip = {"host", "transfer-encoding", "connection"}
        headers = {k: v for k, v in self.headers.items() if k.lower() not in skip}

        try:
            resp = http_requests.request(
                method=method,
                url=target_url,
                headers=headers,
                data=body if body else None,
                stream=True,
                timeout=30,
            )
        except Exception as e:
            self.send_error(502, f"Backend error: {e}")
            return

        # Send response
        self.send_response(resp.status_code)
        skip_resp = {"transfer-encoding", "connection", "content-encoding"}
        for k, v in resp.headers.items():
            if k.lower() not in skip_resp:
                self.send_header(k, v)
        self.end_headers()

        # Stream response body
        try:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    self.wfile.write(chunk)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            resp.close()

    def do_GET(self):
        self._forward_request("GET")

    def do_POST(self):
        self._forward_request("POST")

    def do_PUT(self):
        self._forward_request("PUT")

    def do_DELETE(self):
        self._forward_request("DELETE")

    def do_OPTIONS(self):
        self._forward_request("OPTIONS")

    def do_PATCH(self):
        self._forward_request("PATCH")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    global BACKEND_URL

    parser = argparse.ArgumentParser(
        description="RAG reverse proxy for llama.cpp web UI"
    )
    parser.add_argument(
        "--port", type=int, default=8081,
        help="Port for the RAG proxy (default: 8081)"
    )
    parser.add_argument(
        "--backend", type=str, default="http://127.0.0.1:8080",
        help="llama-server URL (default: http://127.0.0.1:8080)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()
    BACKEND_URL = args.backend.rstrip("/")

    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║     Dharma Scholar — RAG Proxy for llama.cpp UI        ║")
    print("  ╚══════════════════════════════════════════════════════════╝\n")
    print(f"  Backend:  {BACKEND_URL}")
    print(f"  Proxy:    http://{args.host}:{args.port}")
    print()

    # Pre-load RAG so embedding model is ready before first request
    get_rag()

    print(f"\n  Open http://localhost:{args.port} in your browser.\n")

    # Use ThreadingHTTPServer for concurrent requests
    class ThreadedHTTPServer(HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

        def process_request(self, request, client_address):
            """Handle each request in a new thread."""
            t = threading.Thread(target=self._handle, args=(request, client_address))
            t.daemon = True
            t.start()

        def _handle(self, request, client_address):
            try:
                self.finish_request(request, client_address)
            except Exception:
                self.handle_error(request, client_address)
            finally:
                self.shutdown_request(request)

    server = ThreadedHTTPServer((args.host, args.port), RAGProxyHandler)
    print(f"  Serving on http://{args.host}:{args.port} ...")
    print("  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
