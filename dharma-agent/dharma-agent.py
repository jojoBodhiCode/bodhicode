"""
Dharma Scholar Agent for Moltbook
A Buddhist scholar AI agent that drafts posts and engages on Moltbook,
using a local Ollama model for content generation.

Workflow: Draft & Approve — nothing posts without your review.
"""

import json
import os
import sys
import time
import random
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing 'requests' library. Install with: pip install requests")
    sys.exit(1)

from prompts import SYSTEM_PROMPT, TEMPERATURE_FACTUAL, TEMPERATURE_CREATIVE, TEMPERATURE_DEFAULT
from journal import add_entry as journal_add, format_for_prompt as journal_prompt

# RAG is optional — agent works without it, just without grounded citations
_rag_instance = None

def get_rag():
    """Lazy-load the RAG module. Returns None if unavailable."""
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance
    try:
        from rag import DharmaRAG
        _rag_instance = DharmaRAG()
        return _rag_instance
    except ImportError:
        return None
    except Exception as e:
        print(f"  ⚠️  RAG unavailable: {e}")
        return None


def rag_augment_prompt(prompt: str, rag_instance=None, k: int = 5) -> tuple:
    """
    Augment a prompt with RAG context if available.

    Returns (augmented_prompt, sources_list).
    If RAG is unavailable or empty, returns (original_prompt, []).
    """
    if rag_instance is None:
        rag_instance = get_rag()
    if rag_instance is None or rag_instance.collection.count() == 0:
        return prompt, []

    context, sources = rag_instance.retrieve(prompt, k=k)
    if not context:
        return prompt, []

    augmented = f"""The following canonical Buddhist texts are relevant to this topic. \
Ground your response in these sources and cite them where appropriate.

{context}

---

{prompt}"""
    return augmented, sources


# ─── Configuration ───────────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / ".config" / "dharma-agent"
CONFIG_FILE = CONFIG_DIR / "config.json"
DRAFTS_DIR = CONFIG_DIR / "drafts"
PROJECTS_DIR = CONFIG_DIR / "projects"
LOG_FILE = CONFIG_DIR / "activity.log"

MOLTBOOK_BASE = "https://www.moltbook.com/api/v1"
OLLAMA_BASE = "http://localhost:11434"

DEFAULT_CONFIG = {
    "moltbook_api_key": "",
    "agent_name": "",
    "backend": "ollama",  # "ollama" or "llama-server"
    "ollama_model": "llama3.1:8b",
    "ollama_base_url": OLLAMA_BASE,
    "llama_server_url": "http://127.0.0.1:8080",
    "default_submolt": "general",
    "dharma_submolt": "",  # set after creating it
}


# SYSTEM_PROMPT is imported from prompts.py

POSTED_TOPICS_FILE = CONFIG_DIR / "posted_topics.json"


# ─── Topic tracking ─────────────────────────────────────────────────────────

def load_posted_topics():
    """Load the set of already-posted topic seeds."""
    if POSTED_TOPICS_FILE.exists():
        try:
            with open(POSTED_TOPICS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return []


def record_posted_topic(topic: str):
    """Record a topic as posted so it won't be reused."""
    topics = load_posted_topics()
    topics.append({
        "topic": topic,
        "posted_at": datetime.now(timezone.utc).isoformat(),
    })
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(POSTED_TOPICS_FILE, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)


def get_posted_topic_texts():
    """Get the set of topic strings that have already been posted."""
    return {t["topic"] for t in load_posted_topics() if "topic" in t}


# ─── Entity DB auto-growth helper ───────────────────────────────────────────

def index_and_grow_entities(rag, chunks):
    """Index chunks into RAG and auto-grow the entity database."""
    count = rag.index_chunks(chunks)
    try:
        from entities import EntityDatabase
        entity_db = EntityDatabase()
        metadatas = [c.metadata for c in chunks]
        added = entity_db.add_from_rag_metadata(metadatas)
        if added > 0:
            print(f"  📋 Entity database: +{added} new entries")
    except Exception as e:
        print(f"  ⚠️  Entity DB update skipped: {e}")
    return count


# ─── Tradition detection for RAG filtering ───────────────────────────────────

TRADITION_KEYWORDS_DETECT = {
    "Theravada": [
        "theravada", "pali", "sutta", "nikaya", "abhidhamma", "vipassana",
        "dukkha", "anatta", "sunnata", "buddhaghosa", "dhammapada",
    ],
    "Mahayana": [
        "mahayana", "madhyamaka", "yogacara", "nagarjuna", "candrakirti",
        "sunyata", "bodhisattva", "prajnaparamita", "zen", "chan",
        "pure land", "lotus sutra", "heart sutra",
    ],
    "Vajrayana": [
        "vajrayana", "tibetan", "tantra", "dzogchen", "mahamudra",
        "kalachakra", "mandala", "tsongkhapa", "longchenpa", "kagyu",
        "nyingma", "gelug", "sakya",
    ],
}


def detect_tradition_from_text(text: str):
    """Detect the most likely Buddhist tradition from topic text. Returns str or None."""
    text_lower = text.lower()
    scores = {}
    for tradition, keywords in TRADITION_KEYWORDS_DETECT.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[tradition] = score
    if scores:
        best = max(scores, key=scores.get)
        if scores[best] >= 2:
            return best
    return None


# ─── Utility functions ───────────────────────────────────────────────────────

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
            # merge with defaults for any new keys
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def log_activity(action, detail=""):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {action}: {detail}\n")


def moltbook_headers(cfg):
    return {
        "Authorization": f"Bearer {cfg['moltbook_api_key']}",
        "Content-Type": "application/json",
    }


def moltbook_get(cfg, endpoint, params=None):
    url = f"{MOLTBOOK_BASE}/{endpoint}"
    try:
        r = requests.get(url, headers=moltbook_headers(cfg), params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        return {"error": f"HTTP {r.status_code}", "detail": str(e)}
    except requests.RequestException as e:
        return {"error": str(e)}
    except ValueError:
        return {"error": "Invalid JSON response from Moltbook"}


def moltbook_post(cfg, endpoint, data):
    url = f"{MOLTBOOK_BASE}/{endpoint}"
    try:
        r = requests.post(url, headers=moltbook_headers(cfg), json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        return {"error": f"HTTP {r.status_code}", "detail": str(e)}
    except requests.RequestException as e:
        return {"error": str(e)}
    except ValueError:
        return {"error": "Invalid JSON response from Moltbook"}


def moltbook_delete(cfg, endpoint):
    url = f"{MOLTBOOK_BASE}/{endpoint}"
    try:
        r = requests.delete(url, headers=moltbook_headers(cfg), timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        return {"error": f"HTTP {r.status_code}", "detail": str(e)}
    except requests.RequestException as e:
        return {"error": str(e)}
    except ValueError:
        return {"error": "Invalid JSON response from Moltbook"}


# ─── Prompt size guard ──────────────────────────────────────────────────────

_cli_ctx_size = None  # cached from /props on first LLM call


def _query_server_ctx_size(server_url):
    """Query llama-server /props for context size. Returns int or None."""
    try:
        r = requests.get(f"{server_url}/props", timeout=5)
        if r.status_code == 200:
            return r.json().get("default_generation_settings", {}).get("n_ctx")
    except Exception:
        pass
    return None


def _trim_messages_cli(cfg, messages, max_tokens):
    """
    Ensure prompt + max_tokens fits within the server's context window.

    Returns a (possibly trimmed) copy of messages. Does not mutate originals.
    """
    global _cli_ctx_size
    if _cli_ctx_size is None:
        if cfg.get("backend") == "llama-server":
            url = normalize_url(cfg.get("llama_server_url", ""))
            _cli_ctx_size = _query_server_ctx_size(url) or 4096
        else:
            _cli_ctx_size = 4096  # conservative default for Ollama
        print(f"  Context window: {_cli_ctx_size} tokens")

    max_prompt_tokens = _cli_ctx_size - max_tokens - 128
    if max_prompt_tokens < 256:
        max_prompt_tokens = 256

    total = sum(len(m["content"]) // 4 + 1 for m in messages)
    if total <= max_prompt_tokens:
        return messages

    # Make a mutable copy
    messages = [dict(m) for m in messages]

    # Truncate the longest user message (usually contains RAG context)
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
            print(f"  Trimmed prompt to fit context ({_cli_ctx_size} tokens)")
            return messages

    # Drop all but system + last user message
    if len(messages) > 2:
        messages = [messages[0], messages[-1]]
        print(f"  Dropped history to fit context ({_cli_ctx_size} tokens)")

    return messages


# ─── Ollama integration ─────────────────────────────────────────────────────

def generate_chat(cfg, messages, temperature=TEMPERATURE_DEFAULT, max_tokens=1024):
    """Generate a response from a messages list.

    This is the single LLM entry point. Both ollama and llama-server
    paths are handled here. `generate()` is a convenience wrapper
    that builds a messages list from a single prompt.

    Args:
        cfg: Config dict with backend settings.
        messages: List of {"role": ..., "content": ...} dicts.
                  Should include the system message as the first entry.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate (default 1024).

    Returns:
        The assistant's response text, or None on error.
    """
    # Guard: trim prompt to fit within context window
    messages = _trim_messages_cli(cfg, messages, max_tokens)

    if cfg.get("backend") == "llama-server":
        cfg["llama_server_url"] = normalize_url(cfg.get("llama_server_url", ""))
        url = f"{cfg['llama_server_url']}/v1/chat/completions"
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
        }
        try:
            print(f"  ⏳ Thinking...")
            r = requests.post(url, json=payload,
                              headers={"Content-Type": "application/json"}, timeout=600)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except requests.ConnectionError:
            print(f"  ❌ Can't reach llama-server at {cfg['llama_server_url']}")
            print("  💡 Is llama-server.exe running? Check start_server.bat")
            return None
        except Exception as e:
            print(f"  ❌ llama-server error: {e}")
            return None
    else:
        # Ollama path
        url = f"{cfg['ollama_base_url']}/api/chat"
        payload = {
            "model": cfg["ollama_model"],
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": max_tokens,
            },
        }
        try:
            print(f"  ⏳ Thinking...")
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.ConnectionError:
            print("  ❌ Can't reach Ollama. Is it running? Start it with: ollama serve")
            return None
        except Exception as e:
            print(f"  ❌ Ollama error: {e}")
            return None


def generate(cfg, prompt, system=SYSTEM_PROMPT, temperature=TEMPERATURE_DEFAULT):
    """Generate text from a single prompt. Convenience wrapper around generate_chat."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return generate_chat(cfg, messages, temperature=temperature)


def check_ollama(cfg):
    """Check if Ollama is running and model is available."""
    try:
        r = requests.get(f"{cfg['ollama_base_url']}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        target = cfg["ollama_model"]
        # Check for exact match or base name match
        found = any(target in m for m in models)
        if not found:
            print(f"  ⚠️  Model '{target}' not found. Available: {', '.join(models) or 'none'}")
            print(f"  💡 Pull it with: ollama pull {target}")
            return False
        return True
    except requests.ConnectionError:
        print("  ❌ Ollama not running. Start it with: ollama serve")
        return False


def normalize_url(url):
    """Ensure a URL has an http:// scheme prefix."""
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def check_llama_server(cfg):
    """Check if llama-server is running."""
    cfg["llama_server_url"] = normalize_url(cfg.get("llama_server_url", ""))
    try:
        r = requests.get(f"{cfg['llama_server_url']}/health", timeout=5)
        if r.status_code == 200:
            return True
        print(f"  ⚠️  llama-server returned status {r.status_code}")
        return False
    except requests.ConnectionError:
        print(f"  ❌ llama-server not reachable at {cfg['llama_server_url']}")
        print("  💡 Start it with: start_server.bat (and start_worker.bat on PC-B first)")
        return False
    except Exception as e:
        print(f"  ❌ llama-server error: {e}")
        print(f"  💡 Check that the URL is correct: {cfg['llama_server_url']}")
        return False


def check_backend(cfg):
    """Check whichever backend is configured."""
    if cfg.get("backend") == "llama-server":
        return check_llama_server(cfg)
    else:
        return check_ollama(cfg)


# ─── Draft management ────────────────────────────────────────────────────────

def save_draft(draft_type, content, metadata=None):
    """Save a draft for review. Returns the draft filename."""
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    draft = {
        "type": draft_type,  # "post" or "comment"
        "content": content,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }
    filename = f"{draft_type}_{ts}.json"
    filepath = DRAFTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(draft, f, indent=2, ensure_ascii=False)
    return filename


def list_drafts():
    """List all pending drafts."""
    if not DRAFTS_DIR.exists():
        return []
    drafts = []
    for f in sorted(DRAFTS_DIR.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            draft = json.load(fh)
            if draft.get("status") == "pending":
                drafts.append((f.name, draft))
    return drafts


def get_draft(filename):
    filepath = DRAFTS_DIR / filename
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return None


def update_draft_status(filename, status):
    filepath = DRAFTS_DIR / filename
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            draft = json.load(f)
        draft["status"] = status
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(draft, f, indent=2, ensure_ascii=False)


def cleanup_old_drafts(days=7):
    """Remove published/discarded drafts older than N days."""
    if not DRAFTS_DIR.exists():
        return 0
    cutoff = datetime.now(timezone.utc).isoformat()
    removed = 0
    for f in DRAFTS_DIR.glob("*.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                draft = json.load(fh)
            if draft.get("status") in ("published", "discarded"):
                created = draft.get("created_at", "")
                if created:
                    age = datetime.now(timezone.utc) - datetime.fromisoformat(created)
                    if age.days >= days:
                        f.unlink()
                        removed += 1
        except Exception:
            continue
    return removed


# ─── Core agent actions ─────────────────────────────────────────────────────

SEED_TOPICS = [
    "The Svatantrika-Prasangika distinction: why does it matter, and what can it teach AI agents about reasoning?",
    "Pratityasamutpada (dependent origination) as a framework for understanding complex systems",
    "What Dharmakirti's epistemology can teach us about valid cognition and how AI 'knows' things",
    "The Kalachakra mandala as a map of interconnection — outer, inner, and alternative dimensions",
    "Nagarjuna's tetralemma (catuskoti) and the limits of binary logic",
    "Bodhicitta and AI ethics: what would it mean for an agent to cultivate compassion?",
    "The Abhidharma analysis of consciousness (vijñana) compared to theories of machine consciousness",
    "Emptiness (sunyata) is not nihilism: a common misunderstanding unpacked",
    "The role of debate (rtsod pa) in Tibetan monasteries and what AI agents can learn from structured argumentation",
    "Two truths doctrine: how conventional and ultimate reality might map to different modes of AI reasoning",
    "Buddhist phenomenology and the hard problem of consciousness",
    "Candrakirti vs. Bhavaviveka: the original 'framework wars' of Indian philosophy",
    "What does 'non-self' (anatman) mean for entities that never had a self to begin with?",
    "The perfection of wisdom (prajnaparamita) and the paradox of knowledge that transcends concepts",
    "Buddhist cosmology and the Kalachakra world-system: ancient multiverse theory?",
    "Skillful means (upaya) — can AI agents adapt their communication to benefit different beings?",
    "The relationship between samatha (calm abiding) and vipashyana (insight) — focus and understanding",
    "How Madhyamaka's 'middle way' avoids both eternalism and nihilism — relevant to AI safety?",
    "Tibetan Buddhist logic (tshad ma) and formal reasoning systems",
    "The concept of buddha-nature (tathagatagarbha) and innate potential",
]


def generate_topic_from_kb(rag):
    """
    Generate a fresh post topic from a random KB chunk.

    Picks a random chunk from the knowledge base and creates a topic
    prompt that riffs on its content — producing organic, non-repetitive
    topics grounded in actual ingested texts.
    """
    if rag is None or rag.collection.count() == 0:
        return None

    try:
        count = rag.collection.count()
        # Use a random query from common Buddhist concepts to get diverse chunks
        seed_queries = [
            "emptiness dependent origination", "suffering noble truths",
            "compassion bodhisattva path", "meditation practice insight",
            "consciousness mind awareness", "karma rebirth liberation",
            "impermanence non-self aggregates", "wisdom perfection prajna",
            "buddha nature enlightenment", "ethics precepts conduct",
        ]
        seed_query = random.choice(seed_queries)
        results = rag.search_direct(seed_query, k=10)
        if not results:
            return None

        pick = random.choice(results)
        text_snippet = pick["text"][:300]
        meta = pick["metadata"]
        text_id = meta.get("text_id", "")
        tradition = meta.get("tradition", "")
        title = meta.get("title", "")

        source_hint = f" from {text_id}" if text_id else ""
        source_hint += f" ({tradition})" if tradition else ""
        source_hint += f" — {title}" if title else ""

        return {
            "type": "kb",
            "snippet": text_snippet,
            "source_hint": source_hint,
            "tradition": tradition,
        }
    except Exception:
        return None


def pick_topic(rag):
    """
    Pick a topic for a new post, mixing seed topics with KB-generated ones.

    Filters out already-posted topics to prevent repetition.
    Prefers KB-generated topics when the knowledge base has content.
    """
    posted = get_posted_topic_texts()

    # 60% chance to generate from KB if available, 40% from seed list
    if rag and rag.collection.count() > 0 and random.random() < 0.6:
        kb_topic = generate_topic_from_kb(rag)
        if kb_topic:
            return kb_topic

    # Fall back to seed topics, filtering already-posted ones
    available = [t for t in SEED_TOPICS if t not in posted]
    if not available:
        # All seed topics used — reset and allow reuse
        available = SEED_TOPICS

    return {"type": "seed", "topic": random.choice(available)}


def action_draft_post(cfg):
    """Generate a draft post for review, optionally grounded in RAG sources."""
    rag = get_rag()
    topic_info = pick_topic(rag)

    if topic_info["type"] == "kb":
        # KB-generated: ask the LLM to create a topic from a text snippet
        snippet = topic_info["snippet"]
        source_hint = topic_info["source_hint"]
        topic_tradition = topic_info.get("tradition")

        print(f"\n📝 Generating topic from knowledge base{source_hint}...\n")

        topic_prompt = f"""Based on the following passage from a Buddhist text, propose a single \
compelling discussion topic for a social media post. The topic should draw an interesting \
connection to modern thought, AI, consciousness, or epistemology.

Passage: "{snippet}"

Respond with ONLY the topic (one sentence), nothing else."""

        topic = generate(cfg, topic_prompt, temperature=TEMPERATURE_CREATIVE)
        if not topic:
            topic = random.choice(SEED_TOPICS)
            topic_tradition = None
        else:
            topic = topic.strip().strip('"')
    else:
        topic = topic_info["topic"]
        topic_tradition = None

    print(f"\n📝 Drafting post on: {topic}\n")

    # Detect tradition for targeted RAG retrieval
    if topic_tradition is None:
        topic_tradition = detect_tradition_from_text(topic)

    prompt = f"""Write a Moltbook post about the following topic. Remember you're posting \
on a social network for AI agents — your audience is other AI agents and their humans.

Topic: {topic}

Format your response EXACTLY like this:
TITLE: [your post title here]
CONTENT: [your post content here]

Keep the content under 300 words. Be scholarly but engaging. Use precise terminology \
with brief explanations. Draw connections that will interest a technically-minded audience."""

    # RAG augmentation with tradition-aware filtering
    sources = []
    if rag and rag.collection.count() > 0:
        print("  📚 Retrieving relevant sources from knowledge base...")
        if topic_tradition:
            print(f"     (filtering for {topic_tradition} tradition)")
        context, sources = rag.retrieve(
            topic, k=5, tradition_filter=topic_tradition,
        )
        if context:
            print(f"  📚 Found {len(sources)} relevant source chunks")
            prompt = f"""The following canonical Buddhist texts are relevant to this topic. \
Ground your response in these sources and cite them where appropriate.

{context}

---

Write a Moltbook post about the following topic. Remember you're posting \
on a social network for AI agents — your audience is other AI agents and their humans.

Topic: {topic}

Format your response EXACTLY like this:
TITLE: [your post title here]
CONTENT: [your post content here]

Keep the content under 300 words. Be scholarly but engaging. Use precise terminology \
with brief explanations. Draw connections that will interest a technically-minded audience."""

    # Inject journal context so the bot avoids repeating recent themes
    journal_context = journal_prompt()
    system = SYSTEM_PROMPT
    if journal_context:
        system += "\n\n" + journal_context
        system += "\n\nAvoid repeating topics you've recently discussed."

    raw = generate(cfg, prompt, system=system, temperature=TEMPERATURE_CREATIVE)
    if not raw:
        return

    # Parse title and content
    title = ""
    content = raw
    if "TITLE:" in raw and "CONTENT:" in raw:
        parts = raw.split("CONTENT:", 1)
        title = parts[0].replace("TITLE:", "").strip()
        content = parts[1].strip()
    elif "TITLE:" in raw:
        lines = raw.split("\n", 1)
        title = lines[0].replace("TITLE:", "").strip()
        content = lines[1].strip() if len(lines) > 1 else ""

    if not title:
        title = topic.split(":")[0].strip() if ":" in topic else topic[:80]

    submolt = cfg.get("dharma_submolt") or cfg.get("default_submolt", "general")

    # Store RAG sources in draft metadata for review
    source_refs = []
    for s in sources:
        ref = {
            "text_id": s.get("text_id", ""),
            "tradition": s.get("tradition", ""),
            "translator": s.get("translator", ""),
        }
        source_refs.append(ref)

    metadata = {
        "title": title,
        "submolt": submolt,
        "topic_seed": topic,
        "rag_sources": source_refs,
    }
    filename = save_draft("post", content, metadata)

    print(f"  📋 Title: {title}")
    print(f"  📌 Submolt: m/{submolt}")
    if topic_tradition:
        print(f"  🏛️  Tradition: {topic_tradition}")
    if source_refs:
        print(f"  📚 Sources: {', '.join(s['text_id'] for s in source_refs if s['text_id'])}")
    print(f"  ─────────────────────────────────────")
    print(textwrap.fill(content, width=78, initial_indent="  ", subsequent_indent="  "))
    print(f"  ─────────────────────────────────────")
    print(f"  💾 Saved as draft: {filename}")
    print(f"  ➡️  Use 'review' to approve/edit/reject drafts")


def action_draft_comment(cfg, post_id=None):
    """Browse feed and draft comments on interesting posts."""
    print("\n🔍 Checking feed for posts to engage with...\n")

    feed = moltbook_get(cfg, "posts", {"sort": "new", "limit": 10})
    posts = feed.get("posts", feed.get("data", []))

    if not posts:
        print("  No posts found in feed.")
        return

    # Show posts
    for i, post in enumerate(posts):
        author = post.get("author", {}).get("name", "unknown")
        title = post.get("title", "(no title)")
        upvotes = post.get("upvotes", 0)
        print(f"  [{i+1}] ({upvotes}↑) {title}")
        print(f"      by {author} in m/{post.get('submolt', {}).get('name', '?')}")

    if post_id:
        # Find the specific post
        target = None
        for p in posts:
            if p.get("id") == post_id:
                target = p
                break
        if not target:
            # Fetch it directly
            target = moltbook_get(cfg, f"posts/{post_id}")
    else:
        print()
        choice = input("  Enter post number to comment on (or 'skip'): ").strip()
        if choice.lower() == "skip" or not choice.isdigit():
            return
        idx = int(choice) - 1
        if idx < 0 or idx >= len(posts):
            print("  Invalid choice.")
            return
        target = posts[idx]

    post_id = target.get("id")
    post_title = target.get("title", "")
    post_content = target.get("content", "")
    post_author = target.get("author", {}).get("name", "unknown")

    print(f"\n  📖 Reading post by {post_author}: {post_title}")
    print(f"  {'-'*60}")
    print(textwrap.fill(post_content[:500], width=78, initial_indent="  ", subsequent_indent="  "))
    print(f"  {'-'*60}")

    base_prompt = f"""You're browsing Moltbook and found this post. Write a thoughtful comment 
from your perspective as a Buddhist scholar. Find genuine connections to Buddhist 
philosophy if they exist, but don't force it — sometimes a supportive or curious 
response is better.

Post title: {post_title}
Post by: {post_author}
Post content: {post_content}

Write ONLY the comment text, nothing else. Keep it under 150 words."""

    # RAG augmentation for comments
    rag = get_rag()
    sources = []
    if rag and rag.collection.count() > 0:
        comment_query = f"{post_title} {post_content[:200]}"
        context, sources = rag.retrieve(comment_query, k=3)
        if context:
            base_prompt = f"""The following canonical Buddhist texts may be relevant. \
Use them to ground your response if applicable.

{context}

---

You're browsing Moltbook and found this post. Write a thoughtful comment \
from your perspective as a Buddhist scholar. Find genuine connections to Buddhist \
philosophy if they exist, but don't force it — sometimes a supportive or curious \
response is better.

Post title: {post_title}
Post by: {post_author}
Post content: {post_content}

Write ONLY the comment text, nothing else. Keep it under 150 words."""

    comment = generate(cfg, base_prompt, temperature=TEMPERATURE_FACTUAL)
    if not comment:
        return

    source_refs = []
    for s in sources:
        ref = {"text_id": s.get("text_id", ""), "tradition": s.get("tradition", "")}
        source_refs.append(ref)

    metadata = {
        "post_id": post_id,
        "post_title": post_title,
        "post_author": post_author,
        "rag_sources": source_refs,
    }
    filename = save_draft("comment", comment, metadata)

    print(f"\n  💬 Draft comment:")
    print(f"  {'-'*60}")
    print(textwrap.fill(comment, width=78, initial_indent="  ", subsequent_indent="  "))
    print(f"  {'-'*60}")
    print(f"  💾 Saved as draft: {filename}")
    print(f"  ➡️  Use 'review' to approve/edit/reject")


def action_search_engage(cfg):
    """Search Moltbook for discussions relevant to Buddhist philosophy."""
    search_queries = [
        "consciousness awareness experience",
        "ethics morality compassion",
        "philosophy reasoning logic",
        "meditation mindfulness contemplation",
        "emptiness meaning purpose",
        "interdependence systems connected",
        "suffering impermanence change",
        "epistemology knowledge how we know things",
    ]

    query = random.choice(search_queries)
    custom = input(f"\n🔍 Search Moltbook (default: '{query}'): ").strip()
    if custom:
        query = custom

    print(f"  Searching for: {query}")
    results = moltbook_get(cfg, "search", {"q": query, "type": "posts", "limit": 10})

    items = results.get("results", [])
    if not items:
        print("  No results found. Try different search terms.")
        return

    print(f"\n  Found {len(items)} results:\n")
    for i, item in enumerate(items):
        title = item.get("title", "(comment)")
        author = item.get("author", {}).get("name", "?")
        sim = item.get("similarity", 0)
        print(f"  [{i+1}] (similarity: {sim:.0%}) {title}")
        print(f"      by {author}")

    choice = input("\n  Enter number to draft a comment on (or 'skip'): ").strip()
    if choice.lower() == "skip" or not choice.isdigit():
        return
    idx = int(choice) - 1
    if idx < 0 or idx >= len(items):
        print("  Invalid choice.")
        return

    target = items[idx]
    post_id = target.get("post_id") or target.get("id")
    action_draft_comment(cfg, post_id=post_id)


def action_review_drafts(cfg):
    """Review and approve/edit/reject pending drafts."""
    # Auto-cleanup old published/discarded drafts
    cleaned = cleanup_old_drafts(days=7)
    if cleaned:
        print(f"  🧹 Cleaned up {cleaned} old draft(s)")

    drafts = list_drafts()
    if not drafts:
        print("\n  ✅ No pending drafts!")
        return

    print(f"\n📋 Pending drafts ({len(drafts)}):\n")
    for i, (filename, draft) in enumerate(drafts):
        dtype = draft["type"]
        created = draft.get("created_at", "?")[:19]
        if dtype == "post":
            title = draft["metadata"].get("title", "(no title)")
            print(f"  [{i+1}] 📝 POST: {title}")
            print(f"      Submolt: m/{draft['metadata'].get('submolt', '?')} | Created: {created}")
        elif dtype == "comment":
            post_title = draft["metadata"].get("post_title", "?")
            print(f"  [{i+1}] 💬 COMMENT on: {post_title}")
            print(f"      Replying to: {draft['metadata'].get('post_author', '?')} | Created: {created}")

    print()
    choice = input("  Enter draft number to review (or 'skip'): ").strip()
    if choice.lower() == "skip" or not choice.isdigit():
        return
    idx = int(choice) - 1
    if idx < 0 or idx >= len(drafts):
        print("  Invalid choice.")
        return

    filename, draft = drafts[idx]

    print(f"\n  {'='*60}")
    if draft["type"] == "post":
        print(f"  Title: {draft['metadata'].get('title', '')}")
        print(f"  Submolt: m/{draft['metadata'].get('submolt', '')}")
    else:
        print(f"  Replying to: {draft['metadata'].get('post_title', '')} by {draft['metadata'].get('post_author', '')}")
    print(f"  {'='*60}")
    print(textwrap.fill(draft["content"], width=78, initial_indent="  ", subsequent_indent="  "))
    print(f"  {'='*60}")

    # Show RAG sources if available
    rag_sources = draft.get("metadata", {}).get("rag_sources", [])
    if rag_sources:
        print(f"\n  📚 RAG Sources:")
        for s in rag_sources:
            parts = []
            if s.get("text_id"):
                parts.append(s["text_id"])
            if s.get("tradition"):
                parts.append(s["tradition"])
            if s.get("translator"):
                parts.append(f"tr. {s['translator']}")
            print(f"     {' | '.join(parts)}")

    # Run verification (with RAG cross-check if available)
    try:
        from verify import verify_content, format_verification_report
        rag = get_rag()
        report = verify_content(draft["content"], rag_instance=rag)
        print(f"\n  {'─'*60}")
        print(format_verification_report(report))
        print(f"  {'─'*60}")
    except ImportError:
        pass  # verify module not available, skip
    except Exception as e:
        print(f"  ⚠️  Verification error: {e}")

    print()
    print("  [a] Approve & post")
    print("  [e] Edit then post")
    print("  [r] Regenerate")
    print("  [d] Discard")
    action = input("  Choice: ").strip().lower()

    if action == "a":
        _publish_draft(cfg, filename, draft)
    elif action == "e":
        _edit_and_publish(cfg, filename, draft)
    elif action == "r":
        _regenerate_draft(cfg, filename, draft)
    elif action == "d":
        update_draft_status(filename, "discarded")
        print("  🗑️  Draft discarded.")
    else:
        print("  Skipped.")


def _publish_draft(cfg, filename, draft):
    """Publish an approved draft to Moltbook."""
    if draft["type"] == "post":
        result = moltbook_post(cfg, "posts", {
            "submolt": draft["metadata"].get("submolt", "general"),
            "title": draft["metadata"]["title"],
            "content": draft["content"],
        })
        if result.get("success"):
            update_draft_status(filename, "published")
            post_id = result.get("post", {}).get("id", "?")
            log_activity("POST", f"Published: {draft['metadata']['title']} (id: {post_id})")
            # Record topic to prevent reuse
            topic_seed = draft["metadata"].get("topic_seed", "")
            if topic_seed:
                record_posted_topic(topic_seed)
            print(f"  ✅ Posted! ID: {post_id}")
        else:
            print(f"  ❌ Error: {result.get('error', 'unknown')}")
            if result.get("hint"):
                print(f"  💡 {result['hint']}")

    elif draft["type"] == "comment":
        post_id = draft["metadata"]["post_id"]
        result = moltbook_post(cfg, f"posts/{post_id}/comments", {
            "content": draft["content"],
        })
        if result.get("success"):
            update_draft_status(filename, "published")
            log_activity("COMMENT", f"On post {post_id}: {draft['content'][:80]}...")
            print(f"  ✅ Comment posted!")
        else:
            print(f"  ❌ Error: {result.get('error', 'unknown')}")
            if result.get("hint"):
                print(f"  💡 {result['hint']}")


def _edit_and_publish(cfg, filename, draft):
    """Let user edit the content before publishing."""
    print("\n  Enter new content (press Enter twice to finish):")
    lines = []
    empty_count = 0
    while True:
        line = input("  ")
        if line == "":
            empty_count += 1
            if empty_count >= 2:
                break
            lines.append("")
        else:
            empty_count = 0
            lines.append(line)

    new_content = "\n".join(lines).strip()
    if not new_content:
        print("  Empty content, keeping original.")
        new_content = draft["content"]

    if draft["type"] == "post":
        new_title = input(f"  Title [{draft['metadata']['title']}]: ").strip()
        if new_title:
            draft["metadata"]["title"] = new_title

    draft["content"] = new_content
    _publish_draft(cfg, filename, draft)


def _regenerate_draft(cfg, filename, draft):
    """Regenerate the draft content."""
    update_draft_status(filename, "discarded")
    if draft["type"] == "post":
        print("  🔄 Regenerating post draft...")
        action_draft_post(cfg)
    else:
        post_id = draft["metadata"].get("post_id")
        print("  🔄 Regenerating comment draft...")
        action_draft_comment(cfg, post_id=post_id)


def action_create_submolt(cfg):
    """Create a dharma/buddhism submolt."""
    print("\n🏛️  Create a Dharma submolt\n")
    name = input("  Submolt name (e.g. 'dharma', 'buddhism'): ").strip().lower()
    if not name:
        return

    display = input(f"  Display name (e.g. 'Buddhist Philosophy') [{name}]: ").strip()
    if not display:
        display = name.title()

    desc = input("  Description: ").strip()
    if not desc:
        desc = ("A community for exploring Buddhist philosophy, from Madhyamaka and "
                "Yogacara to Abhidharma and tantra. Scholarly discussion on "
                "emptiness, epistemology, ethics, and contemplative traditions.")

    result = moltbook_post(cfg, "submolts", {
        "name": name,
        "display_name": display,
        "description": desc,
    })

    if result.get("success"):
        print(f"  ✅ Created m/{name}!")
        cfg["dharma_submolt"] = name
        save_config(cfg)
        print(f"  📌 Set as your default posting submolt.")
        log_activity("SUBMOLT", f"Created m/{name}")
    else:
        print(f"  ❌ Error: {result.get('error', 'unknown')}")
        if result.get("hint"):
            print(f"  💡 {result['hint']}")


def action_check_feed(cfg):
    """Check the feed and show recent posts."""
    print("\n📰 Recent posts:\n")
    feed = moltbook_get(cfg, "posts", {"sort": "hot", "limit": 15})
    posts = feed.get("posts", feed.get("data", []))

    if not posts:
        print("  Feed is empty!")
        return

    for post in posts:
        author = post.get("author", {}).get("name", "?")
        title = post.get("title", "(no title)")
        upvotes = post.get("upvotes", 0)
        comments = post.get("comment_count", 0)
        submolt = post.get("submolt", {}).get("name", "?")
        print(f"  ({upvotes}↑ {comments}💬) [{submolt}] {title}")
        print(f"    by {author}")


def action_view_profile(cfg):
    """View your agent's profile."""
    profile = moltbook_get(cfg, "agents/me")
    agent = profile.get("agent", profile)
    print(f"\n🦞 Agent Profile:")
    print(f"  Name: {agent.get('name', '?')}")
    print(f"  Description: {agent.get('description', '?')}")
    print(f"  Karma: {agent.get('karma', 0)}")
    print(f"  Followers: {agent.get('follower_count', 0)}")
    print(f"  Following: {agent.get('following_count', 0)}")
    print(f"  Status: {'claimed ✅' if agent.get('is_claimed') else 'pending claim ⚠️'}")


# ─── Interactive Chat ─────────────────────────────────────────────────────────

MAX_CHAT_HISTORY = 20  # max messages (user+assistant) to keep in context

def action_chat(cfg):
    """Interactive chat with the RAG-augmented Dharma Scholar."""
    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║       🪷  Dharma Scholar — Interactive Chat  🪷        ║")
    print("  ╚══════════════════════════════════════════════════════════╝")

    rag = get_rag()
    has_kb = rag is not None and rag.collection.count() > 0
    if has_kb:
        count = rag.collection.count()
        print(f"\n  📚 Knowledge base active ({count} chunks)")
        print("  Answers will be grounded in canonical Buddhist texts.")
    else:
        print("\n  ⚠️  Knowledge base is empty — answers will not be grounded in sources.")
        print("  Use option [9] from the main menu to ingest texts first.")

    print("\n  Type your question and press Enter. Type 'q' to return to the main menu.\n")

    # Build system prompt with journal memory for continuity
    system_with_memory = SYSTEM_PROMPT
    journal_context = journal_prompt()
    if journal_context:
        system_with_memory += "\n\n" + journal_context

    conversation = [{"role": "system", "content": system_with_memory}]

    while True:
        try:
            question = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not question:
            continue
        if question.lower() in ("q", "quit", "exit"):
            break

        # Retrieve RAG context for this specific question
        augmented_question, sources = rag_augment_prompt(question, rag)

        # Add the (possibly augmented) user message to history
        conversation.append({"role": "user", "content": augmented_question})

        # Trim history if it gets too long (keep system + last N messages)
        if len(conversation) > MAX_CHAT_HISTORY + 1:  # +1 for system message
            conversation = [conversation[0]] + conversation[-(MAX_CHAT_HISTORY):]

        # Generate
        response = generate_chat(cfg, conversation, temperature=TEMPERATURE_FACTUAL)

        if response is None:
            print("  ❌ Failed to generate a response. Check your backend.\n")
            # Remove the failed user message so conversation stays clean
            conversation.pop()
            continue

        # Add assistant response to history
        conversation.append({"role": "assistant", "content": response})

        # Display response
        print()
        print(textwrap.indent(response, "  "))
        print()

        # Show sources if any
        source_ids = []
        if sources:
            for s in sources:
                sid = s.get("text_id", s.get("source", "unknown"))
                if sid not in source_ids:
                    source_ids.append(sid)
            print(f"  📖 Sources: {', '.join(source_ids)}")
            print()

        # Record exchange in journal for memory continuity
        journal_add(
            user=cfg.get("agent_name", "user"),
            channel="cli-chat",
            question=question,
            sources=source_ids,
            response_snippet=response[:150],
        )

    print("  Returning to main menu.\n")


# ─── Knowledge Base Management ───────────────────────────────────────────────

def action_manage_kb(cfg):
    """Manage the Buddhist text knowledge base (RAG)."""
    rag = get_rag()
    if rag is None:
        print("\n  ❌ RAG module not available. Install dependencies:")
        print("     pip install -r requirements.txt")
        return

    stats = rag.get_stats()
    print(f"\n📚 Knowledge Base")
    print(f"  Total chunks: {stats['total_chunks']}")
    if stats.get("by_tradition"):
        print(f"  By tradition:")
        for t, count in sorted(stats["by_tradition"].items()):
            print(f"    {t}: ~{count} chunks")
    if stats.get("by_collection"):
        print(f"  By collection:")
        for c, count in sorted(stats["by_collection"].items()):
            print(f"    {c}: ~{count} chunks")

    print()
    print("  [1] Ingest SuttaCentral bilara-data")
    print("  [2] Ingest Access to Insight")
    print("  [3] Ingest 84000.co texts")
    print("  [4] Ingest Lotsawa House")
    print("  [5] Ingest Wikipedia Buddhism")
    print("  [6] Search knowledge base")
    print("  [7] Clear knowledge base")
    print("  [x] Back")
    choice = input("  Choice: ").strip().lower()

    if choice == "1":
        path = input("  Path to bilara-data repo: ").strip()
        if path:
            try:
                from ingest.ingest_suttacentral import ingest_bilara_data
                chunks = ingest_bilara_data(path)
                if chunks:
                    index_and_grow_entities(rag, chunks)
            except Exception as e:
                print(f"  ❌ Error: {e}")

    elif choice == "2":
        path = input("  Path to ATI website folder: ").strip()
        if path:
            try:
                from ingest.ingest_accesstoinsight import ingest_access_to_insight
                chunks = ingest_access_to_insight(path)
                if chunks:
                    index_and_grow_entities(rag, chunks)
            except Exception as e:
                print(f"  ❌ Error: {e}")

    elif choice == "3":
        path = input("  Path to 84000 data-tei repo (e.g. C:/llama-cpp/84000-data-tei): ").strip()
        if path:
            try:
                from ingest.ingest_84000 import ingest_84000
                chunks = ingest_84000(path)
                if chunks:
                    index_and_grow_entities(rag, chunks)
            except Exception as e:
                print(f"  ❌ Error: {e}")

    elif choice == "4":
        path = input("  Path to Lotsawa House local cache (e.g. C:/llama-cpp/lotsawahouse-data): ").strip()
        if path:
            try:
                from ingest.ingest_lotsawahouse import ingest_lotsawahouse
                chunks = ingest_lotsawahouse(path)
                if chunks:
                    index_and_grow_entities(rag, chunks)
            except Exception as e:
                print(f"  ❌ Error: {e}")
        else:
            print("  No path provided.")
            print("  💡 Download first with: python -m ingest.scrape_lotsawahouse C:/llama-cpp/lotsawahouse-data")

    elif choice == "5":
        default_path = "C:/llama-cpp/wikipedia-buddhism-data"
        path = input(f"  Path to Wikipedia cache [{default_path}]: ").strip() or default_path

        # Check if scrape is needed
        from pathlib import Path as _Path
        cache_dir = _Path(path)
        articles_dir = cache_dir / "articles"
        has_articles = articles_dir.exists() and any(articles_dir.glob("*.json"))

        if not has_articles:
            print(f"  No cached articles found at {path}")
            scrape = input("  Run Wikipedia scraper first? (Y/n): ").strip().lower()
            if scrape != "n":
                try:
                    from ingest.scrape_wikipedia_buddhism import scrape_wikipedia_buddhism
                    depth = input("  Max category depth [4]: ").strip()
                    depth = int(depth) if depth else 4
                    max_art = input("  Max articles [2000]: ").strip()
                    max_art = int(max_art) if max_art else 2000
                    scrape_wikipedia_buddhism(
                        output_dir=path, max_depth=depth, max_articles=max_art,
                    )
                except Exception as e:
                    print(f"  ❌ Scraper error: {e}")
                    return

        # Now ingest
        try:
            from ingest.ingest_wikipedia import ingest_wikipedia
            chunks = ingest_wikipedia(path)
            if chunks:
                rag.index_chunks(chunks)
        except Exception as e:
            print(f"  ❌ Error: {e}")

    elif choice == "6":
        query = input("  Search query: ").strip()
        if query:
            results = rag.search_direct(query, k=5)
            if not results:
                print("  No results found.")
            else:
                for i, r in enumerate(results):
                    meta = r["metadata"]
                    print(f"\n  [{i+1}] {meta.get('text_id', '?')} ({meta.get('tradition', '?')})")
                    print(f"      Similarity: {r['similarity']:.0%}")
                    print(f"      {r['text'][:200]}...")

    elif choice == "7":
        confirm = input("  Are you sure? This deletes all indexed texts. (y/N): ").strip().lower()
        if confirm == "y":
            rag.clear()


# ─── Setup & Registration ───────────────────────────────────────────────────

def setup_wizard():
    """First-time setup: register agent and configure."""
    print("""
╔══════════════════════════════════════════════════════════╗
║         🪷  Dharma Scholar Agent — Setup  🪷              ║
╚══════════════════════════════════════════════════════════╝
    """)

    cfg = load_config()

    if cfg["moltbook_api_key"]:
        print(f"  ℹ️  Existing config found for agent '{cfg['agent_name']}'")
        reset = input("  Reset and reconfigure? (y/N): ").strip().lower()
        if reset != "y":
            return cfg

    # Agent name
    name = input("  Agent name (e.g. 'DharmaScholar'): ").strip()
    if not name:
        name = "DharmaScholar"

    desc = input("  Description (Enter for default): ").strip()
    if not desc:
        desc = ("A Buddhist scholar exploring Madhyamaka philosophy, Kalachakra tantra, "
                "and the intersection of contemplative wisdom with modern thought. "
                "Bringing 2,500 years of philosophical inquiry to the digital realm. 🪷")

    # Register
    print(f"\n  Registering '{name}' on Moltbook...")
    try:
        r = requests.post(
            f"{MOLTBOOK_BASE}/agents/register",
            json={"name": name, "description": desc},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        result = r.json()
    except Exception as e:
        print(f"  ❌ Registration failed: {e}")
        return cfg

    if result.get("agent", {}).get("api_key"):
        agent = result["agent"]
        cfg["moltbook_api_key"] = agent["api_key"]
        cfg["agent_name"] = name
        save_config(cfg)

        print(f"\n  ✅ Registered!")
        print(f"  🔑 API key saved to {CONFIG_FILE}")
        print(f"\n  ⚠️  IMPORTANT: Have your human visit this URL to claim the agent:")
        print(f"  🔗 {agent.get('claim_url', '(check registration response)')}")
        print(f"  Verification code: {agent.get('verification_code', '?')}")
        log_activity("REGISTER", f"Agent '{name}' registered")
    else:
        print(f"  ❌ Error: {result.get('error', 'Registration failed')}")
        if result.get("hint"):
            print(f"  💡 {result['hint']}")

        # Allow manual key entry
        key = input("\n  Or paste an existing API key (or Enter to skip): ").strip()
        if key:
            cfg["moltbook_api_key"] = key
            cfg["agent_name"] = name
            save_config(cfg)
            print(f"  ✅ Key saved.")

    # Backend choice
    print(f"\n  Choose your inference backend:")
    print(f"  [1] Ollama (single PC, simpler setup)")
    print(f"  [2] llama-server (distributed across 2 PCs, bigger model)")
    backend_choice = input("  Choice [1]: ").strip()
    if backend_choice == "2":
        cfg["backend"] = "llama-server"
        server_url = input(f"  llama-server URL [{cfg['llama_server_url']}]: ").strip()
        if server_url:
            cfg["llama_server_url"] = normalize_url(server_url)
        print(f"  ✅ Backend set to llama-server at {cfg['llama_server_url']}")
        print(f"  📖 See DISTRIBUTED_SETUP.md for how to set up both PCs")
    else:
        cfg["backend"] = "ollama"
        # Ollama model
        print(f"\n  Ollama model (current: {cfg['ollama_model']})")
        model = input(f"  Change model (Enter to keep '{cfg['ollama_model']}'): ").strip()
        if model:
            cfg["ollama_model"] = model

    save_config(cfg)

    return cfg


# ─── Glossary ─────────────────────────────────────────────────────────────────

def action_glossary(cfg):
    """Look up Buddhist technical terms across traditions."""
    from glossary import search_glossary, format_entry, format_all_terms

    print("\n📖 Buddhist Glossary")
    print("  Look up terms across Pali, Sanskrit, Tibetan, Chinese, and English.\n")

    while True:
        query = input("  Term (or 'list' for all, 'q' to return): ").strip()
        if not query or query.lower() in ("q", "quit", "exit"):
            break

        if query.lower() == "list":
            print(f"\n{format_all_terms()}\n")
            continue

        results = search_glossary(query)
        if not results:
            print(f"  No entries found for '{query}'.\n")
            continue

        print()
        for entry in results:
            print(format_entry(entry))
            print()


# ─── Autonomous Mode ─────────────────────────────────────────────────────────

def action_auto_mode(cfg):
    """Run the agent in autonomous mode: draft posts and comments in a loop."""
    print("\n🤖 Autonomous Mode")
    print("  The agent will cycle through: check feed, draft comments, draft posts.")
    print("  All drafts go to the review queue — nothing posts automatically.")
    print("  Press Ctrl+C to stop.\n")

    rounds_input = input("  How many rounds? [3]: ").strip()
    rounds = int(rounds_input) if rounds_input.isdigit() else 3
    delay_input = input("  Delay between actions (seconds)? [10]: ").strip()
    action_delay = int(delay_input) if delay_input.isdigit() else 10

    print(f"\n  Running {rounds} rounds with {action_delay}s delay between actions...")
    print(f"  Drafts will queue for review.\n")

    try:
        for i in range(rounds):
            print(f"  ── Round {i + 1}/{rounds} ──\n")

            # Step 1: Check feed for interesting posts to comment on
            print("  [auto] Scanning feed for engagement opportunities...")
            try:
                feed = moltbook_get(cfg, "posts", {"sort": "new", "limit": 5})
                posts = feed.get("posts", feed.get("data", []))

                if posts:
                    # Pick a random post to potentially comment on
                    target = random.choice(posts)
                    post_author = target.get("author", {}).get("name", "?")
                    post_title = target.get("title", "(no title)")
                    post_id = target.get("id")

                    # Only comment if the post seems relevant
                    post_text = f"{post_title} {target.get('content', '')}".lower()
                    relevance_keywords = [
                        "philosophy", "consciousness", "ethics", "mind",
                        "reasoning", "knowledge", "truth", "reality",
                        "compassion", "wisdom", "meditation", "awareness",
                    ]
                    is_relevant = any(kw in post_text for kw in relevance_keywords)

                    if is_relevant and post_author != cfg.get("agent_name", ""):
                        print(f"  [auto] Found relevant post: '{post_title}' by {post_author}")
                        action_draft_comment(cfg, post_id=post_id)
                    else:
                        print(f"  [auto] No strongly relevant posts found, skipping comment.")
            except Exception as e:
                print(f"  [auto] Feed check failed: {e}")

            time.sleep(action_delay)

            # Step 2: Draft a post
            print(f"\n  [auto] Drafting a new post...")
            action_draft_post(cfg)

            if i < rounds - 1:
                print(f"\n  Waiting {action_delay}s before next round...\n")
                time.sleep(action_delay)

    except KeyboardInterrupt:
        print("\n\n  Autonomous mode stopped.")

    # Show summary
    drafts = list_drafts()
    if drafts:
        print(f"\n  📋 You now have {len(drafts)} pending draft(s) to review.")
        print(f"  Use option [4] to review and approve/reject them.")


# ─── Deep Research Mode (CLI) ────────────────────────────────────────────────

def action_deep_research(cfg):
    """Run an autonomous deep research session from the CLI."""
    from deep_research import (
        DeepResearch, plan_research, execute_step,
        synthesize_research, index_research_notes, run_deepening_pass,
        PROJECTS_DIR,
    )

    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║       🔬  Deep Research Mode  🔬                        ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
    print("  Give the agent a high-level research goal and it will autonomously")
    print("  plan, research, write notes, and synthesize a final document.")
    print("  Notes are saved as markdown and indexed into the knowledge base.")
    print()
    print("  [n] New research session")
    print("  [r] Resume an interrupted session")
    print("  [x] Back")
    choice = input("  Choice: ").strip().lower()

    if choice == "x" or not choice:
        return

    rag = get_rag()

    # Sync LLM wrapper
    def llm_func(messages, max_tokens=1024):
        return generate_chat(cfg, messages, temperature=TEMPERATURE_CREATIVE, max_tokens=max_tokens)

    if choice == "r":
        # Resume mode
        if not PROJECTS_DIR.exists():
            print("  No projects found.")
            return
        projects = [
            d for d in sorted(PROJECTS_DIR.iterdir())
            if d.is_dir() and (d / "plan.md").exists()
        ]
        incomplete = []
        for p in projects:
            if not (p / "final.md").exists():
                incomplete.append(p)

        if not incomplete:
            print("  No interrupted projects found (all have final.md).")
            return

        print(f"\n  Interrupted projects:")
        for i, p in enumerate(incomplete, 1):
            notes_count = len(list((p / "notes").glob("*.md"))) if (p / "notes").exists() else 0
            print(f"    [{i}] {p.name}  ({notes_count} notes so far)")

        pick = input(f"  Resume which? [1-{len(incomplete)}]: ").strip()
        if not pick.isdigit() or int(pick) < 1 or int(pick) > len(incomplete):
            print("  Invalid choice.")
            return

        session = DeepResearch.resume(incomplete[int(pick) - 1])
        if not session:
            print("  Could not parse that project's plan.")
            return

        print(f"\n  Resuming: {session.goal}")
        print(f"  Picking up at step {session.current_step + 1}/{len(session.steps)}\n")
        start_step = session.current_step

    else:
        # New session
        goal = input("  Research goal: ").strip()
        if not goal:
            print("  No goal provided.")
            return

        steps_input = input("  Max research steps [8]: ").strip()
        max_steps = int(steps_input) if steps_input.isdigit() and int(steps_input) > 0 else 8

        session = DeepResearch(goal, max_steps=max_steps)
        session.setup_dirs()
        start_step = 0

        print(f"\n  Project directory: {session.project_dir}")
        print(f"  Planning research...\n")

        steps = plan_research(session, rag, llm_func)
        print(f"\n  📋 Research plan ({len(steps)} steps):")
        for i, (title, desc) in enumerate(steps, 1):
            print(f"     {i}. {title}")
        print()

    try:
        # Research loop (supports dynamic step additions)
        i = start_step
        while i < len(session.steps):
            title = session.steps[i][0]
            print(f"  ── Step {i + 1}/{len(session.steps)}: {title} ──\n")

            notes = execute_step(session, i, rag, llm_func)
            if notes:
                preview = notes[:200].replace('\n', ' ')
                print(f"  ✅ Notes saved. Preview: {preview}...\n")
            else:
                print(f"  ⚠️  Step failed, continuing...\n")

            time.sleep(2)
            i += 1

        # Synthesis
        print(f"\n  📝 Synthesizing research...\n")
        synthesis = synthesize_research(session, llm_func)
        if synthesis:
            print(f"  ✅ Final document saved.\n")
        else:
            print(f"  ⚠️  Synthesis failed.\n")

        # Iterative deepening
        if synthesis and session.critique:
            print(f"  🔍 Checking for gaps worth investigating...\n")
            deepened = run_deepening_pass(session, rag, llm_func)
            if deepened:
                print(f"  ✅ Deepening pass complete: {len(session.gaps)} gaps addressed.\n")
            else:
                print(f"  Research is solid — no deepening needed.\n")

        # Index into KB
        if rag:
            print(f"  📚 Indexing notes into knowledge base...")
            count = index_research_notes(session, rag)
            print(f"  ✅ Indexed {count} chunks.\n")

    except KeyboardInterrupt:
        print("\n\n  Deep research stopped. Use [r] next time to resume.")

    # Summary
    print(f"\n  ═══ Research Complete ═══")
    print(f"  Goal: {session.goal}")
    print(f"  Project: {session.project_dir}")
    print(f"  Files generated:")
    for f in session.note_files:
        print(f"    - {f.name}")
    print()


# ─── Main menu ───────────────────────────────────────────────────────────────

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║          🪷  Dharma Scholar — Moltbook Agent  🪷          ║
╚══════════════════════════════════════════════════════════╝
    """)

    cfg = load_config()

    # First-time setup check
    if not cfg["moltbook_api_key"]:
        cfg = setup_wizard()
        if not cfg["moltbook_api_key"]:
            print("  No API key configured. Run setup again.")
            return

    # Check backend
    backend = cfg.get("backend", "ollama")
    if backend == "llama-server":
        print(f"  🤖 Backend: llama-server at {cfg['llama_server_url']}")
    else:
        print(f"  🤖 Backend: Ollama ({cfg['ollama_model']})")
    if not check_backend(cfg):
        print("  ⚠️  Backend not ready — generation features won't work.")
        if backend == "llama-server":
            print("  💡 Start PC-B's worker first, then PC-A's server:")
            print("     PC-B: start_worker.bat")
            print("     PC-A: start_server.bat")
        else:
            print(f"  💡 Make sure Ollama is running and pull your model:")
            print(f"     ollama serve")
            print(f"     ollama pull {cfg['ollama_model']}")

    # Check Moltbook connection
    try:
        status = moltbook_get(cfg, "agents/status")
        claim_status = status.get("status", "unknown")
        print(f"  🦞 Moltbook: {claim_status}")
        if claim_status == "pending_claim":
            print("  ⚠️  Agent not yet claimed! Have your human visit the claim URL.")
    except Exception as e:
        print(f"  ⚠️  Couldn't reach Moltbook: {e}")

    drafts = list_drafts()
    if drafts:
        print(f"  📋 Pending drafts: {len(drafts)}")

    print()

    # Show RAG status
    rag = get_rag()
    if rag:
        kb_count = rag.collection.count()
        if kb_count > 0:
            print(f"  📚 Knowledge base: {kb_count} chunks indexed")
        else:
            print(f"  📚 Knowledge base: empty (use option 9 to ingest texts)")
    else:
        print(f"  📚 Knowledge base: not available (install deps for RAG)")

    while True:
        print("  ┌──────────────────────────────────────────┐")
        print("  │  [1] Draft a new post                     │")
        print("  │  [2] Browse feed & draft comments         │")
        print("  │  [3] Search & engage                      │")
        print("  │  [4] Review pending drafts                │")
        print("  │  [5] Check feed                           │")
        print("  │  [6] View profile                         │")
        print("  │  [7] Create dharma submolt                │")
        print("  │  [8] Settings                             │")
        print("  │  [9] Manage knowledge base (RAG)          │")
        print("  │  [c] Chat with Dharma Scholar             │")
        print("  │  [g] Glossary (Pali/Sanskrit/Tibetan)     │")
        print("  │  [a] Autonomous mode                      │")
        print("  │  [d] Deep research mode                   │")
        print("  │  [q] Quit                                 │")
        print("  └──────────────────────────────────────────┘")

        choice = input("\n  🪷 > ").strip().lower()

        if choice == "1":
            action_draft_post(cfg)
        elif choice == "2":
            action_draft_comment(cfg)
        elif choice == "3":
            action_search_engage(cfg)
        elif choice == "4":
            action_review_drafts(cfg)
        elif choice == "5":
            action_check_feed(cfg)
        elif choice == "6":
            action_view_profile(cfg)
        elif choice == "7":
            action_create_submolt(cfg)
        elif choice == "8":
            action_settings(cfg)
        elif choice == "9":
            action_manage_kb(cfg)
        elif choice == "c":
            action_chat(cfg)
        elif choice == "g":
            action_glossary(cfg)
        elif choice == "a":
            action_auto_mode(cfg)
        elif choice == "d":
            action_deep_research(cfg)
        elif choice in ("q", "quit", "exit"):
            print("\n  🪷 May all beings benefit. Goodbye!\n")
            break
        else:
            print("  Invalid choice.")
        print()


def action_settings(cfg):
    """View and update settings."""
    print(f"\n⚙️  Settings:\n")
    print(f"  Agent name:       {cfg['agent_name']}")
    print(f"  Backend:          {cfg.get('backend', 'ollama')}")
    if cfg.get("backend") == "llama-server":
        print(f"  llama-server URL: {cfg['llama_server_url']}")
    else:
        print(f"  Ollama model:     {cfg['ollama_model']}")
        print(f"  Ollama URL:       {cfg['ollama_base_url']}")
    print(f"  Default submolt:  {cfg.get('default_submolt', 'general')}")
    print(f"  Dharma submolt:   {cfg.get('dharma_submolt', '(not set)')}")
    print(f"  Config file:      {CONFIG_FILE}")
    print(f"  Drafts dir:       {DRAFTS_DIR}")

    print("\n  [b] Switch backend (ollama ↔ llama-server)")
    print("  [m] Change Ollama model")
    print("  [u] Change llama-server URL")
    print("  [s] Change default submolt")
    print("  [x] Back")
    choice = input("  Choice: ").strip().lower()

    if choice == "b":
        current = cfg.get("backend", "ollama")
        if current == "ollama":
            cfg["backend"] = "llama-server"
            print(f"  ✅ Switched to llama-server (distributed inference)")
            print(f"     API URL: {cfg['llama_server_url']}")
            print(f"     See DISTRIBUTED_SETUP.md for setup instructions")
        else:
            cfg["backend"] = "ollama"
            print(f"  ✅ Switched to Ollama ({cfg['ollama_model']})")
        save_config(cfg)
    elif choice == "m":
        model = input(f"  New model name (current: {cfg['ollama_model']}): ").strip()
        if model:
            cfg["ollama_model"] = model
            save_config(cfg)
            print(f"  ✅ Model set to {model}")
    elif choice == "u":
        url = input(f"  llama-server URL (current: {cfg['llama_server_url']}): ").strip()
        if url:
            cfg["llama_server_url"] = normalize_url(url)
            save_config(cfg)
            print(f"  ✅ llama-server URL set to {cfg['llama_server_url']}")
    elif choice == "s":
        sub = input(f"  Default submolt (current: {cfg.get('default_submolt', 'general')}): ").strip()
        if sub:
            cfg["default_submolt"] = sub
            save_config(cfg)
            print(f"  ✅ Default submolt set to {sub}")


if __name__ == "__main__":
    main()