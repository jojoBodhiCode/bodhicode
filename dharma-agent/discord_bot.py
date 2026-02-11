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

import asyncio

from prompts import SYSTEM_PROMPT, TEMPERATURE_FACTUAL, TEMPERATURE_CREATIVE
from journal import add_entry as journal_add, format_for_prompt as journal_prompt
from deep_research import (
    DeepResearch, plan_research, execute_step,
    synthesize_research, index_research_notes, PROJECTS_DIR,
)

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG_FILE = Path.home() / ".config" / "dharma-agent" / "config.json"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
MAX_HISTORY = 2           # max messages per channel (keep small to fit 4096 context)
DISCORD_CHAR_LIMIT = 2000  # Discord message character limit

# ─── LLM lock & research state ──────────────────────────────────────────────

llm_lock = asyncio.Lock()
active_research = None     # DeepResearch instance or None
research_task = None       # asyncio.Task or None
research_channel = None    # Discord channel for research updates


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
        context, sources = rag.retrieve(query, k=3)
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

def generate_response(messages, max_tokens=768, temperature=TEMPERATURE_FACTUAL):
    """Send messages to llama-server and get a response."""
    url = get_llama_url()
    parsed = urlparse(url)

    payload = json.dumps({
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
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
        raw = resp.read()
        conn.close()
        data = json.loads(raw)

        if "choices" not in data:
            # llama-server returned an error (likely context overflow)
            error_msg = data.get("error", {})
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(data))
            print(f"  [LLM] Server error: {error_msg}")
            return None

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


# ─── Async LLM helpers ───────────────────────────────────────────────────────

async def generate_with_lock(messages, max_tokens=768):
    """Async LLM wrapper that serializes access via the lock."""
    async with llm_lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: generate_response(messages, max_tokens, TEMPERATURE_CREATIVE)
        )


async def run_research_background(goal, channel):
    """Run deep research as a background asyncio task, posting updates to Discord."""
    global active_research

    rag = get_rag()
    session = DeepResearch(goal)
    session.setup_dirs()
    active_research = session

    try:
        await channel.send(
            f"**Deep Research Started**\n"
            f"Goal: {goal}\n"
            f"Project: `{session.project_dir}`\n"
            f"Planning research steps..."
        )

        # Planning phase — llm_func for deep_research uses the lock
        def sync_llm(messages, max_tokens=1024):
            """Blocking LLM call used inside run_in_executor."""
            return generate_response(messages, max_tokens, TEMPERATURE_CREATIVE)

        async def async_llm(messages, max_tokens=1024):
            """Async LLM wrapper with lock for research steps."""
            async with llm_lock:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: sync_llm(messages, max_tokens)
                )

        # Plan — run in executor since plan_research is sync
        loop = asyncio.get_event_loop()

        # We need to make deep_research functions work with our async llm.
        # Since they expect a sync callable, we'll run each phase with
        # the lock held, then release between phases.
        async with llm_lock:
            steps = await loop.run_in_executor(
                None, lambda: plan_research(session, rag, sync_llm)
            )

        plan_summary = "\n".join(
            f"{i+1}. **{t}**" for i, (t, _) in enumerate(steps)
        )
        await channel.send(f"**Research Plan** ({len(steps)} steps):\n{plan_summary}")

        # Research loop
        for i in range(len(steps)):
            if session.status == "stopped":
                await channel.send("Research stopped by user.")
                break

            await asyncio.sleep(2)  # yield to Discord event loop

            title = steps[i][0]
            await channel.send(f"Step {i+1}/{len(steps)}: **{title}**...")

            async with llm_lock:
                notes = await loop.run_in_executor(
                    None, lambda idx=i: execute_step(session, idx, rag, sync_llm)
                )

            if notes:
                preview = notes[:300].replace('\n', ' ')
                await channel.send(f"Step {i+1} complete. Preview: {preview[:200]}...")
            else:
                await channel.send(f"Step {i+1} had an issue, continuing...")

        # Synthesis
        if session.status != "stopped":
            await channel.send("Synthesizing all research into a final document...")
            async with llm_lock:
                synthesis = await loop.run_in_executor(
                    None, lambda: synthesize_research(session, sync_llm)
                )

            if synthesis:
                await channel.send(
                    f"Final document saved: `{session.project_dir / 'final.md'}`"
                )
            else:
                await channel.send("Synthesis had an issue.")

            # Index into KB
            if rag:
                count = await loop.run_in_executor(
                    None, lambda: index_research_notes(session, rag)
                )
                await channel.send(f"Indexed {count} chunks into the knowledge base.")

        # Summary
        files_list = "\n".join(f"  - `{f.name}`" for f in session.note_files)
        await channel.send(
            f"**Research Complete**\n"
            f"Project: `{session.project_dir}`\n"
            f"Files:\n{files_list}"
        )

    except asyncio.CancelledError:
        await channel.send("Research task was cancelled.")
    except Exception as e:
        await channel.send(f"Research error: {e}")
        print(f"  [Research] Error: {e}")
    finally:
        active_research = None


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
    global research_task, research_channel

    # Don't respond to ourselves
    if message.author == client.user:
        return

    # Respond in DMs (no mention needed) or when @mentioned in a server
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = any(u.id == client.user.id for u in message.mentions)

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

    # ─── Handle !research commands ───────────────────────────────────────
    if question.startswith("!research"):
        parts = question.split(None, 1)
        subcommand = parts[1] if len(parts) > 1 else ""

        if subcommand.lower() == "status":
            if active_research:
                step_info = (
                    f"Step {active_research.current_step + 1}/{len(active_research.steps)}"
                    if active_research.steps else "planning"
                )
                await message.reply(
                    f"**Research Status:** {active_research.status}\n"
                    f"**Goal:** {active_research.goal}\n"
                    f"**Progress:** {step_info}\n"
                    f"**Project:** `{active_research.project_dir}`"
                )
            else:
                await message.reply("No active research session.")
            return

        if subcommand.lower() == "stop":
            if active_research:
                active_research.status = "stopped"
                await message.reply("Stopping research after current step completes...")
            else:
                await message.reply("No active research to stop.")
            return

        # Start new research: !research <goal>
        if not subcommand:
            await message.reply(
                "**Deep Research Commands:**\n"
                "`!research <goal>` — Start a research session\n"
                "`!research status` — Check progress\n"
                "`!research stop` — Stop current session"
            )
            return

        if active_research:
            await message.reply(
                "A research session is already running. "
                "Use `!research stop` to cancel it first."
            )
            return

        # Launch research as a background task
        research_channel = message.channel
        research_task = asyncio.create_task(
            run_research_background(subcommand, message.channel)
        )
        return

    # ─── Regular message handling ────────────────────────────────────────

    channel_name = "DM" if is_dm else f"#{message.channel.name}"
    print(f"  [Discord] {channel_name} | {message.author}: {question[:80]}...")

    # Show typing indicator while we generate
    async with message.channel.typing():
        # RAG retrieval (run in executor to avoid blocking the event loop)
        loop = asyncio.get_event_loop()
        context, sources = await loop.run_in_executor(
            None, rag_retrieve, question
        )
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

        # Build full messages list with system prompt + journal memory
        journal_section = await loop.run_in_executor(None, journal_prompt)
        if journal_section:
            system_with_memory = f"{SYSTEM_PROMPT}\n\n{journal_section}"
        else:
            system_with_memory = SYSTEM_PROMPT
        messages = [{"role": "system", "content": system_with_memory}] + history

        # Generate response — use the lock to serialize with research
        async with llm_lock:
            response = await loop.run_in_executor(
                None, generate_response, messages
            )

            # If context overflow, retry with no history
            if response is None and len(history) > 1:
                print("  [Discord] Retrying without conversation history...")
                history[:] = [{"role": "user", "content": augmented_question}]
                messages = [{"role": "system", "content": system_with_memory}] + history
                response = await loop.run_in_executor(
                    None, generate_response, messages
                )

        if response is None:
            await message.reply(
                "I'm having trouble generating a response. "
                "The question may be too long, or the backend may be down."
            )
            history.pop()
            return

        # Add assistant response to history (use clean version without RAG context)
        history.append({"role": "assistant", "content": response})

        # Log to journal for persistent memory (non-blocking)
        await loop.run_in_executor(
            None, lambda: journal_add(
                user=str(message.author),
                channel=channel_name,
                question=question,
                sources=source_ids if source_ids else None,
                response_snippet=response[:150],
            )
        )

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
