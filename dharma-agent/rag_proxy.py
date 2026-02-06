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

try:
    from flask import Flask, request, Response
except ImportError:
    print("Missing 'flask'. Install with: pip install flask")
    sys.exit(1)

try:
    import requests as http_requests
except ImportError:
    print("Missing 'requests'. Install with: pip install requests")
    sys.exit(1)

from prompts import SYSTEM_PROMPT

# ─── RAG setup (lazy-loaded) ─────────────────────────────────────────────────

_rag_instance = None


def get_rag():
    """Lazy-load the RAG module. Returns None if unavailable."""
    global _rag_instance
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

    # 1. Ensure system prompt is present
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        # Replace the first system message with our Dharma Scholar prompt
        # (the llama.cpp UI sends a default "You are a helpful assistant" system msg)
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
                messages = list(messages)  # copy
                messages[last_user_idx] = {
                    "role": "user",
                    "content": augmented_content,
                }

    return messages, sources


# ─── Flask proxy ──────────────────────────────────────────────────────────────

app = Flask(__name__)
BACKEND_URL = "http://127.0.0.1:8080"  # overridden by --backend arg


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Intercept chat completions: inject RAG context, then forward."""
    body = request.get_json(force=True)
    messages = body.get("messages", [])

    # Augment with system prompt + RAG
    augmented_messages, sources = augment_messages(messages)
    body["messages"] = augmented_messages

    # Log what we retrieved
    if sources:
        source_ids = []
        for s in sources:
            sid = s.get("text_id", s.get("source", "?"))
            if sid not in source_ids:
                source_ids.append(sid)
        print(f"  [RAG] Sources: {', '.join(source_ids)}")
    else:
        print("  [RAG] No sources retrieved for this query.")

    # Check if streaming is requested
    is_streaming = body.get("stream", False)

    if is_streaming:
        # Stream the response back chunk-by-chunk
        resp = http_requests.post(
            f"{BACKEND_URL}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=600,
        )
        return Response(
            resp.iter_content(chunk_size=None),
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "text/event-stream"),
        )
    else:
        # Non-streaming: forward and return
        resp = http_requests.post(
            f"{BACKEND_URL}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=600,
        )
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def proxy_all(path):
    """Forward all other requests to llama-server unchanged."""
    target_url = f"{BACKEND_URL}/{path}"

    # Forward with same method, headers, query params, and body
    resp = http_requests.request(
        method=request.method,
        url=target_url,
        headers={k: v for k, v in request.headers if k.lower() != "host"},
        params=request.args,
        data=request.get_data(),
        stream=True,
        timeout=30,
    )

    # Build excluded headers (hop-by-hop)
    excluded = {"transfer-encoding", "connection", "content-encoding", "content-length"}
    headers = [(k, v) for k, v in resp.raw.headers.items() if k.lower() not in excluded]

    return Response(
        resp.iter_content(chunk_size=None),
        status=resp.status_code,
        headers=headers,
        content_type=resp.headers.get("Content-Type"),
    )


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

    # Eagerly load RAG so the embedding model is ready before first request
    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║     Dharma Scholar — RAG Proxy for llama.cpp UI        ║")
    print("  ╚══════════════════════════════════════════════════════════╝\n")
    print(f"  Backend:  {BACKEND_URL}")
    print(f"  Proxy:    http://{args.host}:{args.port}")
    print()

    get_rag()  # pre-load embedding model

    print(f"\n  Open http://localhost:{args.port} in your browser.\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
