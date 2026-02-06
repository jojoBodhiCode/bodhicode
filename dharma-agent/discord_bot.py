"""
Discord Bot for Dharma Scholar

A Discord bot that responds when @mentioned, using the RAG pipeline
to ground responses in canonical Buddhist texts.

Setup:
  1. Create a bot at https://discord.com/developers/applications
  2. Enable Message Content Intent under Bot > Privileged Gateway Intents
  3. Invite with OAuth2 URL (scopes: bot; permissions: Send Messages, Read Message History)
  4. Set your token via env var or config:
       set DISCORD_BOT_TOKEN=your-token-here
       python discord_bot.py

The bot will respond only when @mentioned. It maintains per-channel
conversation history for follow-up questions.
"""

import http.client
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

try:
    import discord
except ImportError:
    print("Missing 'discord.py'. Install with: pip install discord.py")
    sys.exit(1)

from prompts import SYSTEM_PROMPT, TEMPERATURE_FACTUAL

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG_FILE = Path.home() / ".config" / "dharma-agent" / "config.json"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
MAX_HISTORY = 20          # max messages per channel to keep in memory
DISCORD_CHAR_LIMIT = 2000  # Discord message character limit


def load_config():
    """Load config from the dharma-agent config file."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def get_bot_token():
    """Get the Discord bot token from env var or config file."""
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if token:
        return token
    cfg = load_config()
    token = cfg.get("discord_bot_token")
    if token:
        return token
    print("  ERROR: No Discord bot token found.")
    print("  Set it via environment variable:")
    print("    set DISCORD_BOT_TOKEN=your-token-here")
    print("  Or add 'discord_bot_token' to ~/.config/dharma-agent/config.json")
    sys.exit(1)


def get_llama_url():
    """Get the llama-server URL from config or default."""
    cfg = load_config()
    url = cfg.get("llama_server_url", LLAMA_SERVER_URL)
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


# ─── RAG setup (lazy-loaded) ─────────────────────────────────────────────────

_rag_instance = None


def get_rag():
    """Lazy-load the RAG module."""
    global _rag_instance
    if _rag_instance is None:
        try:
            from rag import DharmaRAG
            _rag_instance = DharmaRAG()
            count = _rag_instance.collection.count()
            if count > 0:
                print(f"  [RAG] Knowledge base loaded: {count} chunks")
            else:
                print("  [RAG] Knowledge base is empty.")
        except Exception as e:
            print(f"  [RAG] Could not load: {e}")
            return None
    return _rag_instance


def rag_retrieve(query):
    """Retrieve RAG context for a query. Returns (context_str, sources_list)."""
    rag = get_rag()
    if rag is None or rag.collection.count() == 0:
        return "", []
    try:
        context, sources = rag.retrieve(query, k=5)
        return context, sources
    except Exception as e:
        print(f"  [RAG] Retrieval error: {e}")
        return "", []


def format_source_ids(sources):
    """Extract unique source IDs from metadata."""
    ids = []
    for s in sources:
        sid = s.get("text_id", s.get("source", ""))
        if sid and sid not in ids:
            ids.append(sid)
    return ids


# ─── LLM generation ──────────────────────────────────────────────────────────

def generate_response(messages):
    """Send messages to llama-server and get a response."""
    url = get_llama_url()
    parsed = urlparse(url)

    payload = json.dumps({
        "messages": messages,
        "temperature": TEMPERATURE_FACTUAL,
        "top_p": 0.9,
        "max_tokens": 1024,
        "stream": False,
    }).encode("utf-8")

    try:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=600)
        conn.request(
            "POST", "/v1/chat/completions",
            body=payload,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
        )
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [LLM] Generation error: {e}")
        return None


# ─── Message splitting ────────────────────────────────────────────────────────

def split_message(text, limit=DISCORD_CHAR_LIMIT):
    """Split a long message into chunks that fit Discord's character limit."""
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break

        # Try to split at a paragraph break
        split_at = text.rfind("\n\n", 0, limit)
        if split_at == -1:
            # Try a single newline
            split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            # Try a sentence break
            split_at = text.rfind(". ", 0, limit)
            if split_at != -1:
                split_at += 1  # include the period
        if split_at == -1:
            # Hard split
            split_at = limit

        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()

    return chunks


# ─── Discord Bot ──────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Per-channel conversation history: {channel_id: [{"role": ..., "content": ...}]}
channel_history = defaultdict(list)


@client.event
async def on_ready():
    print(f"\n  Discord bot logged in as {client.user}")
    print(f"  DM me directly, or @mention me in a server.")
    print(f"  LLM backend: {get_llama_url()}")
    # Pre-load RAG
    get_rag()
    print("  Ready!\n")


@client.event
async def on_message(message):
    # Don't respond to ourselves
    if message.author == client.user:
        return

    # Respond in DMs (no mention needed) or when @mentioned in a server
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = client.user in message.mentions

    if not is_dm and not is_mentioned:
        return

    # Strip the mention from the message to get the actual question
    question = message.content
    if not is_dm:
        for mention in message.mentions:
            question = question.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
    question = question.strip()

    if not question:
        await message.reply("Ask me anything about Buddhism!")
        return

    channel_name = "DM" if is_dm else f"#{message.channel.name}"
    print(f"  [Discord] {channel_name} | {message.author}: {question[:80]}...")

    # Show typing indicator while we generate
    async with message.channel.typing():
        # RAG retrieval
        context, sources = rag_retrieve(question)
        source_ids = format_source_ids(sources)

        if source_ids:
            print(f"  [RAG] Sources: {', '.join(source_ids)}")

        # Build augmented user message
        if context:
            augmented_question = (
                "The following canonical Buddhist texts are relevant to this topic. "
                "Ground your response in these sources and cite them where appropriate.\n\n"
                f"{context}\n\n---\n\n{question}"
            )
        else:
            augmented_question = question

        # Get channel history and add new message
        history = channel_history[message.channel.id]
        history.append({"role": "user", "content": augmented_question})

        # Trim history (keep last N messages)
        if len(history) > MAX_HISTORY:
            history[:] = history[-MAX_HISTORY:]

        # Build full messages list with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        # Generate response (this blocks for 30-60s, typing indicator stays active)
        import asyncio
        response = await asyncio.get_event_loop().run_in_executor(
            None, generate_response, messages
        )

        if response is None:
            await message.reply(
                "I'm having trouble connecting to my LLM backend. "
                "Please make sure llama-server is running."
            )
            # Remove the failed user message from history
            history.pop()
            return

        # Add assistant response to history (use clean version without RAG context)
        history.append({"role": "assistant", "content": response})

        # Append source citations
        if source_ids:
            source_footer = f"\n\n*Sources: {', '.join(source_ids)}*"
        else:
            source_footer = ""

        full_response = response + source_footer

        # Split and send
        chunks = split_message(full_response)
        for i, chunk in enumerate(chunks):
            if i == 0:
                await message.reply(chunk)
            else:
                await message.channel.send(chunk)

    print(f"  [Discord] Response sent ({len(response)} chars)")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║     Dharma Scholar — Discord Bot                       ║")
    print("  ╚══════════════════════════════════════════════════════════╝\n")

    token = get_bot_token()
    print(f"  LLM backend: {get_llama_url()}")

    client.run(token, log_handler=None)


if __name__ == "__main__":
    main()
