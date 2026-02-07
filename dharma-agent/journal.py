"""
Persistent journal for the Dharma Scholar bot.

Gives the bot a sense of continuity across restarts by recording
brief entries about each conversation exchange. Recent entries are
injected into the system prompt so the bot remembers what it's been
doing and who it's talked to.

Storage: ~/.config/dharma-agent/journal.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


JOURNAL_PATH = Path.home() / ".config" / "dharma-agent" / "journal.json"
MAX_ENTRIES = 200       # total entries to keep on disk
PROMPT_ENTRIES = 10     # entries to inject into system prompt


def load_journal() -> List[dict]:
    """Load journal entries from disk."""
    if not JOURNAL_PATH.exists():
        return []
    try:
        with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f)
        return entries if isinstance(entries, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_journal(entries: List[dict]):
    """Save journal entries to disk, trimming to MAX_ENTRIES."""
    entries = entries[-MAX_ENTRIES:]
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def add_entry(
    user: str,
    channel: str,
    question: str,
    sources: Optional[List[str]] = None,
    response_snippet: str = "",
):
    """
    Record a journal entry after a conversation exchange.

    Args:
        user: Discord username
        channel: Channel name or "DM"
        question: The user's question (truncated)
        sources: List of source IDs used in the response
        response_snippet: First ~100 chars of the bot's response
    """
    entries = load_journal()
    entry = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "user": str(user),
        "channel": channel,
        "topic": question[:120],
        "sources": sources or [],
        "response": response_snippet[:150],
    }
    entries.append(entry)
    save_journal(entries)


def format_for_prompt() -> str:
    """
    Format recent journal entries for injection into the system prompt.

    Returns a concise summary the bot can reference for continuity.
    """
    entries = load_journal()
    if not entries:
        return ""

    recent = entries[-PROMPT_ENTRIES:]

    lines = ["## Recent Memory\n"]
    lines.append(
        "These are your recent conversations. You can reference them "
        "naturally if relevant (e.g., 'as we discussed earlier...'). "
        "Don't force references — only mention them when genuinely relevant.\n"
    )

    for e in recent:
        sources_str = f" (sources: {', '.join(e['sources'])})" if e.get("sources") else ""
        lines.append(
            f"- [{e['time']}] {e['user']} in {e['channel']}: "
            f"\"{e['topic']}\"{sources_str}"
        )

    return "\n".join(lines)
