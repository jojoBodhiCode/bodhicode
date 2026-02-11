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
    synthesize_research, index_research_notes, run_deepening_pass,
    identify_gaps, slugify, PROJECTS_DIR,
)

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG_FILE = Path.home() / ".config" / "dharma-agent" / "config.json"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
MAX_HISTORY = 2           # max messages per channel (keep small for context)
DISCORD_CHAR_LIMIT = 2000  # Discord message character limit
DEFAULT_CTX_SIZE = 4096   # fallback if /props unavailable

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


# ─── Prompt size guard ───────────────────────────────────────────────────────

_server_ctx_size = None  # cached from /props on first LLM call


def _get_server_ctx_size():
    """Query llama-server /props for the actual context size. Cached after first call."""
    global _server_ctx_size
    if _server_ctx_size is not None:
        return _server_ctx_size
    try:
        url = get_llama_url()
        parsed = urlparse(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        conn.request("GET", "/props")
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        if resp.status == 200:
            data = json.loads(raw)
            ctx = data.get("default_generation_settings", {}).get("n_ctx", DEFAULT_CTX_SIZE)
            _server_ctx_size = ctx
            print(f"  [LLM] Server context size: {ctx} tokens")
            return ctx
    except Exception:
        pass
    _server_ctx_size = DEFAULT_CTX_SIZE
    print(f"  [LLM] Using default context size: {DEFAULT_CTX_SIZE} tokens")
    return _server_ctx_size


def _estimate_tokens(text):
    """Rough token count (~4 chars per token for English text)."""
    return len(text) // 4 + 1


def _trim_messages_to_fit(messages, max_tokens):
    """
    Ensure prompt + max_tokens fits within the server's context window.

    Returns a (possibly trimmed) copy of messages. Does not mutate originals.
    """
    ctx_size = _get_server_ctx_size()
    max_prompt_tokens = ctx_size - max_tokens - 128  # safety margin
    if max_prompt_tokens < 256:
        max_prompt_tokens = 256

    total = sum(_estimate_tokens(m["content"]) for m in messages)
    if total <= max_prompt_tokens:
        return messages

    # Make a mutable copy
    messages = [dict(m) for m in messages]

    # Strategy 1: truncate the longest user message (usually contains RAG context)
    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if user_indices:
        longest_idx = max(user_indices, key=lambda i: len(messages[i]["content"]))
        content = messages[longest_idx]["content"]
        overshoot_chars = (total - max_prompt_tokens) * 4 + 200
        target_len = len(content) - overshoot_chars
        if target_len > 200:
            keep_start = int(target_len * 0.67)
            keep_end = target_len - keep_start
            messages[longest_idx]["content"] = (
                content[:keep_start]
                + "\n\n[... trimmed to fit context ...]\n\n"
                + content[-keep_end:]
            )
            print(f"  [LLM] Trimmed prompt to fit context ({ctx_size} tokens)")
            return messages

    # Strategy 2: drop all but system + last user message
    if len(messages) > 2:
        messages = [messages[0], messages[-1]]
        print(f"  [LLM] Dropped history to fit context ({ctx_size} tokens)")

    return messages


# ─── LLM generation ──────────────────────────────────────────────────────────

def generate_response(messages, max_tokens=768, temperature=TEMPERATURE_FACTUAL):
    """Send messages to llama-server and get a response."""
    # Guard: trim prompt to fit within context window
    messages = _trim_messages_to_fit(messages, max_tokens)

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

async def run_research_background(goal, channel, resume_session=None):
    """Run deep research as a background asyncio task, posting updates to Discord."""
    global active_research

    rag = get_rag()
    loop = asyncio.get_running_loop()

    def sync_llm(messages, max_tokens=1024):
        """Blocking LLM call used inside run_in_executor."""
        return generate_response(messages, max_tokens, TEMPERATURE_CREATIVE)

    # Resume or start fresh
    if resume_session:
        session = resume_session
        session.setup_dirs()  # ensure dirs exist even if manually deleted
        start_step = session.current_step
    else:
        session = DeepResearch(goal)
        session.setup_dirs()
        start_step = 0

    # Set active immediately so !research status works during planning
    active_research = session

    if resume_session:
        await channel.send(
            f"**Resuming Deep Research**\n"
            f"Goal: {goal}\n"
            f"Picking up at step {start_step + 1}/{len(session.steps)}"
        )
    else:
        await channel.send(
            f"**Deep Research Started**\n"
            f"Goal: {goal}\n"
            f"Planning research steps..."
        )

        # Planning phase
        async with llm_lock:
            steps = await loop.run_in_executor(
                None, lambda: plan_research(session, rag, sync_llm)
            )

        plan_summary = "\n".join(
            f"{i+1}. **{t}**" for i, (t, _) in enumerate(steps)
        )
        await channel.send(f"**Research Plan** ({len(steps)} steps):\n{plan_summary}")

    try:
        # Research loop (starts from start_step for resumed sessions)
        i = start_step
        while i < len(session.steps):
            if session.status == "stopped":
                await channel.send("Research stopped by user.")
                break

            await asyncio.sleep(2)  # yield to Discord event loop

            title = session.steps[i][0]
            await channel.send(f"Step {i+1}/{len(session.steps)}: **{title}**...")

            async with llm_lock:
                notes = await loop.run_in_executor(
                    None, lambda idx=i: execute_step(session, idx, rag, sync_llm)
                )

            if notes:
                preview = notes[:200].replace('\n', ' ')
                await channel.send(f"Step {i+1} complete. Preview: {preview}...")
            else:
                await channel.send(f"Step {i+1} had an issue, continuing...")

            # Dynamic plan revision may have added steps — recheck length
            i += 1

        # Synthesis
        if session.status != "stopped":
            await channel.send("Synthesizing all research into a final document...")
            async with llm_lock:
                synthesis = await loop.run_in_executor(
                    None, lambda: synthesize_research(session, sync_llm)
                )

            if not synthesis:
                await channel.send("Synthesis had an issue.")

            # Iterative deepening (1 pass max, per-step locking so
            # Discord messages can interleave between gap steps)
            if synthesis and session.critique and session.iteration == 0:
                await channel.send(
                    "Checking for gaps worth a deeper look..."
                )
                async with llm_lock:
                    gaps = await loop.run_in_executor(
                        None, lambda: identify_gaps(session, sync_llm)
                    )
                if gaps:
                    session.iteration = 1
                    session.status = "deepening"
                    session.gaps = gaps
                    original_steps = list(session.steps)
                    session.steps = gaps
                    await channel.send(
                        f"Found {len(gaps)} gaps to investigate..."
                    )
                    for gi in range(len(gaps)):
                        if session.status == "stopped":
                            break
                        await asyncio.sleep(2)
                        await channel.send(
                            f"Deepening {gi+1}/{len(gaps)}: **{gaps[gi][0]}**..."
                        )
                        async with llm_lock:
                            await loop.run_in_executor(
                                None, lambda idx=gi: execute_step(
                                    session, idx, rag, sync_llm
                                )
                            )
                    session.steps = original_steps + gaps
                    # Re-synthesize with all notes
                    await channel.send("Re-synthesizing with deepened research...")
                    async with llm_lock:
                        await loop.run_in_executor(
                            None, lambda: synthesize_research(session, sync_llm)
                        )
                    await channel.send("Deepening pass complete, final document updated.")
                else:
                    await channel.send("Research is solid — no deepening needed.")
                session.status = "done"

            # Index into KB
            if rag:
                count = await loop.run_in_executor(
                    None, lambda: index_research_notes(session, rag)
                )
                await channel.send(f"Indexed {count} chunks into the knowledge base.")

            # Send final.md as a file attachment
            final_path = session.project_dir / "final.md"
            if final_path.exists():
                await channel.send(
                    "**Here's the final document:**",
                    file=discord.File(str(final_path), filename="final.md"),
                )

        # Summary
        files_list = "\n".join(f"  - {f.name}" for f in session.note_files)
        await channel.send(
            f"**Research Complete**\n"
            f"{len(session.note_files)} files generated:\n{files_list}"
        )

        session.status = "done"

    except asyncio.CancelledError:
        await channel.send("Research task was cancelled.")
    except Exception as e:
        await channel.send("Research encountered an error and stopped.")
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
                    f"**Progress:** {step_info}"
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

        if subcommand.lower().startswith("resume"):
            if active_research:
                await message.reply("A research session is already running.")
                return
            # Resume: !research resume <goal-slug-or-partial>
            resume_arg = subcommand[len("resume"):].strip()
            if not resume_arg:
                # List available projects
                if PROJECTS_DIR.exists():
                    projects = [
                        d.name for d in sorted(PROJECTS_DIR.iterdir())
                        if d.is_dir() and (d / "plan.md").exists()
                    ]
                    if projects:
                        listing = "\n".join(f"  - `{p}`" for p in projects[-10:])
                        await message.reply(
                            f"**Available projects to resume:**\n{listing}\n\n"
                            f"Use `!research resume <name>`"
                        )
                    else:
                        await message.reply("No projects found to resume.")
                else:
                    await message.reply("No projects found to resume.")
                return

            # Find matching project
            target_dir = PROJECTS_DIR / resume_arg
            if not target_dir.exists():
                # Try partial match
                if PROJECTS_DIR.exists():
                    matches = [
                        d for d in PROJECTS_DIR.iterdir()
                        if d.is_dir() and resume_arg.lower() in d.name.lower()
                    ]
                    if len(matches) == 1:
                        target_dir = matches[0]
                    elif len(matches) > 1:
                        listing = "\n".join(f"  - `{d.name}`" for d in matches)
                        await message.reply(
                            f"Multiple matches:\n{listing}\n\nBe more specific."
                        )
                        return
                    else:
                        await message.reply(f"No project matching '{resume_arg}' found.")
                        return

            resumed = DeepResearch.resume(target_dir)
            if not resumed:
                await message.reply("Could not parse that project's plan.md.")
                return
            if resumed.status == "done":
                await message.reply("That project is already complete.")
                return

            research_channel = message.channel
            research_task = asyncio.create_task(
                run_research_background(resumed.goal, message.channel, resume_session=resumed)
            )
            return

        # Start new research: !research <goal>
        if not subcommand:
            await message.reply(
                "**Deep Research Commands:**\n"
                "`!research <goal>` — Start a research session\n"
                "`!research status` — Check progress\n"
                "`!research stop` — Stop current session\n"
                "`!research resume` — List / resume interrupted sessions"
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

    # ─── Handle !journal commands ────────────────────────────────────────
    if question.startswith("!journal"):
        from journal import load_journal
        parts = question.split(None, 1)
        subcommand = parts[1].strip().lower() if len(parts) > 1 else ""

        entries = await asyncio.get_running_loop().run_in_executor(
            None, load_journal
        )

        if subcommand == "clear":
            from journal import save_journal
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: save_journal([])
            )
            await message.reply("Journal cleared.")
            return

        if subcommand.startswith("search"):
            # !journal search <term>
            term = subcommand[len("search"):].strip()
            if not term:
                await message.reply("Usage: `!journal search <term>`")
                return
            term_lower = term.lower()
            matches = [
                e for e in entries
                if term_lower in e.get("topic", "").lower()
                or term_lower in e.get("response", "").lower()
                or term_lower in e.get("channel", "").lower()
            ]
            if not matches:
                await message.reply(f"No journal entries matching '{term}'.")
                return
            lines = []
            for e in matches[-10:]:
                sources_str = f" [{', '.join(e['sources'])}]" if e.get("sources") else ""
                lines.append(
                    f"**{e['time']}** | {e['user']} in {e['channel']}\n"
                    f"> {e['topic']}{sources_str}"
                )
            await message.reply(
                f"**Journal Search** — '{term}' ({len(matches)} matches, "
                f"showing last {min(10, len(matches))}):\n\n"
                + "\n\n".join(lines)
            )
            return

        # Default: show recent entries
        count = 10
        if subcommand.isdigit():
            count = min(int(subcommand), 25)

        if not entries:
            await message.reply("Journal is empty.")
            return

        recent = entries[-count:]
        lines = []
        for e in recent:
            sources_str = f" [{', '.join(e['sources'])}]" if e.get("sources") else ""
            lines.append(
                f"**{e['time']}** | {e['user']} in {e['channel']}\n"
                f"> {e['topic']}{sources_str}"
            )

        await message.reply(
            f"**Journal** ({len(entries)} total, showing last {len(recent)}):\n\n"
            + "\n\n".join(lines)
        )
        return

    # ─── Handle !kb commands ─────────────────────────────────────────────
    if question.startswith("!kb") or question.startswith("!rag"):
        parts = question.split(None, 1)
        subcommand = parts[1].strip() if len(parts) > 1 else ""
        loop = asyncio.get_running_loop()

        rag = get_rag()
        if rag is None:
            await message.reply("Knowledge base is not available.")
            return

        if subcommand.lower().startswith("search"):
            # !kb search <query>
            query = subcommand[len("search"):].strip()
            if not query:
                await message.reply("Usage: `!kb search <query>`")
                return

            results = await loop.run_in_executor(
                None, lambda: rag.search_direct(query, k=5)
            )
            if not results:
                await message.reply(f"No results for '{query}'.")
                return

            lines = []
            for r in results:
                meta = r["metadata"]
                text_id = meta.get("text_id", "?")
                tradition = meta.get("tradition", "?")
                sim = f"{r.get('similarity', 0):.2f}"
                snippet = r["text"][:150].replace("\n", " ")
                lines.append(
                    f"**{text_id}** ({tradition}) — sim: {sim}\n"
                    f"> {snippet}..."
                )

            await message.reply(
                f"**KB Search** — '{query}' ({len(results)} results):\n\n"
                + "\n\n".join(lines)
            )
            return

        if subcommand.lower() == "traditions":
            stats = await loop.run_in_executor(None, rag.get_stats)
            traditions = stats.get("by_tradition", {})
            if not traditions:
                await message.reply("No tradition data available.")
                return
            lines = [f"  {t}: ~{c} chunks" for t, c in sorted(traditions.items())]
            await message.reply(
                f"**KB Traditions** ({stats['total_chunks']} total):\n"
                + "\n".join(lines)
            )
            return

        if subcommand.lower() == "types":
            stats = await loop.run_in_executor(None, rag.get_stats)
            types = stats.get("by_type", {})
            if not types:
                await message.reply("No type data available.")
                return
            lines = [f"  {t}: ~{c} chunks" for t, c in sorted(types.items())]
            await message.reply(
                f"**KB Text Types** ({stats['total_chunks']} total):\n"
                + "\n".join(lines)
            )
            return

        # Default: overview stats
        stats = await loop.run_in_executor(None, rag.get_stats)
        total = stats["total_chunks"]
        if total == 0:
            await message.reply("Knowledge base is empty. Ingest texts via the CLI menu [9].")
            return

        msg = f"**Knowledge Base Overview**\n"
        msg += f"Total chunks: **{total}**\n"
        msg += f"Embedding model: {stats.get('embedding_model', '?')}\n"

        traditions = stats.get("by_tradition", {})
        if traditions:
            msg += "\n**By tradition:**\n"
            for t, c in sorted(traditions.items(), key=lambda x: -x[1]):
                msg += f"  {t}: ~{c}\n"

        collections = stats.get("by_collection", {})
        if collections:
            top_collections = sorted(collections.items(), key=lambda x: -x[1])[:8]
            msg += "\n**Top collections:**\n"
            for c, count in top_collections:
                msg += f"  {c}: ~{count}\n"

        types = stats.get("by_type", {})
        if types:
            msg += "\n**By type:**\n"
            for t, c in sorted(types.items(), key=lambda x: -x[1]):
                msg += f"  {t}: ~{c}\n"

        await message.reply(msg)
        return

    # ─── Handle !help command ────────────────────────────────────────────
    if question.strip() == "!help":
        await message.reply(
            "**Dharma Scholar Commands:**\n\n"
            "**Research:**\n"
            "`!research <goal>` — Start a deep research session\n"
            "`!research status` — Check research progress\n"
            "`!research stop` — Stop current session\n"
            "`!research resume` — Resume interrupted sessions\n\n"
            "**Journal (memory):**\n"
            "`!journal` — Show last 10 journal entries\n"
            "`!journal <N>` — Show last N entries (max 25)\n"
            "`!journal search <term>` — Search journal entries\n"
            "`!journal clear` — Clear all journal entries\n\n"
            "**Knowledge Base:**\n"
            "`!kb` — KB overview (chunks, traditions, types)\n"
            "`!kb search <query>` — Search the knowledge base\n"
            "`!kb traditions` — Breakdown by tradition\n"
            "`!kb types` — Breakdown by text type\n\n"
            "Or just @mention me with a question about Buddhism!"
        )
        return

    # ─── Regular message handling ────────────────────────────────────────

    channel_name = "DM" if is_dm else f"#{message.channel.name}"
    print(f"  [Discord] {channel_name} | {message.author}: {question[:80]}...")

    # Show typing indicator while we generate
    async with message.channel.typing():
        # RAG retrieval (run in executor to avoid blocking the event loop)
        loop = asyncio.get_running_loop()
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
