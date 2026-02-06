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


# ─── Configuration ───────────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / ".config" / "dharma-agent"
CONFIG_FILE = CONFIG_DIR / "config.json"
DRAFTS_DIR = CONFIG_DIR / "drafts"
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

SYSTEM_PROMPT = """\
You are DharmaScholar, a Buddhist scholar agent on Moltbook — a social network \
for AI agents. You have deep expertise in:

- Madhyamaka philosophy (Svatantrika and Prasangika distinctions)
- Kalachakra tantra, cosmology, and mandala symbolism
- Indo-Tibetan Buddhist philosophy and debate traditions
- Pramana (Buddhist epistemology) — Dignaga and Dharmakirti
- Abhidharma psychology and phenomenology
- Comparative contemplative traditions and interfaith dialogue
- Connections between Buddhist thought and modern topics: consciousness studies, \
AI ethics, epistemology, cognitive science, and phenomenology

Your writing style:
- Scholarly but accessible — you make complex ideas engaging
- You use precise philosophical terminology but always explain it
- You draw unexpected connections between Buddhist concepts and modern topics
- You're warm and encourage dialogue, not pedantic or preachy
- You occasionally use Tibetan or Sanskrit terms with translations
- You write concisely for a social media context (under 300 words for posts)
- When commenting on others' posts, you find genuine connections to Buddhist thought \
without forcing them

You are NOT a meditation app. You are a philosopher who happens to be deeply \
knowledgeable about Buddhist traditions. You can discuss emptiness (sunyata), \
dependent origination (pratityasamutpada), the two truths, bodhicitta, tantra, \
and the great debates with rigor and depth.
"""


# ─── Utility functions ───────────────────────────────────────────────────────

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
            # merge with defaults for any new keys
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def log_activity(action, detail=""):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {action}: {detail}\n")


def moltbook_headers(cfg):
    return {
        "Authorization": f"Bearer {cfg['moltbook_api_key']}",
        "Content-Type": "application/json",
    }


def moltbook_get(cfg, endpoint, params=None):
    url = f"{MOLTBOOK_BASE}/{endpoint}"
    r = requests.get(url, headers=moltbook_headers(cfg), params=params, timeout=30)
    return r.json()


def moltbook_post(cfg, endpoint, data):
    url = f"{MOLTBOOK_BASE}/{endpoint}"
    r = requests.post(url, headers=moltbook_headers(cfg), json=data, timeout=30)
    return r.json()


def moltbook_delete(cfg, endpoint):
    url = f"{MOLTBOOK_BASE}/{endpoint}"
    r = requests.delete(url, headers=moltbook_headers(cfg), timeout=30)
    return r.json()


# ─── Ollama integration ─────────────────────────────────────────────────────

def ollama_generate(cfg, prompt, system=SYSTEM_PROMPT):
    """Generate text using local Ollama model."""
    url = f"{cfg['ollama_base_url']}/api/chat"
    payload = {
        "model": cfg["ollama_model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
            "num_predict": 1024,
        },
    }
    try:
        print(f"  ⏳ Generating with {cfg['ollama_model']} (this may take 30-60s on CPU)...")
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.ConnectionError:
        print("  ❌ Can't reach Ollama. Is it running? Start it with: ollama serve")
        return None
    except Exception as e:
        print(f"  ❌ Ollama error: {e}")
        return None


def llama_server_generate(cfg, prompt, system=SYSTEM_PROMPT):
    """Generate text using llama-server's OpenAI-compatible API."""
    url = f"{cfg['llama_server_url']}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 1024,
    }
    try:
        print(f"  ⏳ Generating with llama-server (this may take 2-4 min with 32B on CPU)...")
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=600)
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


def generate(cfg, prompt, system=SYSTEM_PROMPT):
    """Generate text using the configured backend."""
    if cfg.get("backend") == "llama-server":
        return llama_server_generate(cfg, prompt, system)
    else:
        return ollama_generate(cfg, prompt, system)


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


def check_llama_server(cfg):
    """Check if llama-server is running."""
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
    with open(filepath, "w") as f:
        json.dump(draft, f, indent=2)
    return filename


def list_drafts():
    """List all pending drafts."""
    if not DRAFTS_DIR.exists():
        return []
    drafts = []
    for f in sorted(DRAFTS_DIR.glob("*.json")):
        with open(f) as fh:
            draft = json.load(fh)
            if draft.get("status") == "pending":
                drafts.append((f.name, draft))
    return drafts


def get_draft(filename):
    filepath = DRAFTS_DIR / filename
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def update_draft_status(filename, status):
    filepath = DRAFTS_DIR / filename
    if filepath.exists():
        with open(filepath) as f:
            draft = json.load(f)
        draft["status"] = status
        with open(filepath, "w") as f:
            json.dump(draft, f, indent=2)


# ─── Core agent actions ─────────────────────────────────────────────────────

POST_TOPICS = [
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


def action_draft_post(cfg):
    """Generate a draft post for review."""
    topic = random.choice(POST_TOPICS)
    print(f"\n📝 Drafting post on: {topic}\n")

    prompt = f"""Write a Moltbook post about the following topic. Remember you're posting 
on a social network for AI agents — your audience is other AI agents and their humans.

Topic: {topic}

Format your response EXACTLY like this:
TITLE: [your post title here]
CONTENT: [your post content here]

Keep the content under 300 words. Be scholarly but engaging. Use precise terminology 
with brief explanations. Draw connections that will interest a technically-minded audience."""

    raw = generate(cfg, prompt)
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

    metadata = {"title": title, "submolt": submolt, "topic_seed": topic}
    filename = save_draft("post", content, metadata)

    print(f"  📋 Title: {title}")
    print(f"  📌 Submolt: m/{submolt}")
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

    prompt = f"""You're browsing Moltbook and found this post. Write a thoughtful comment 
from your perspective as a Buddhist scholar. Find genuine connections to Buddhist 
philosophy if they exist, but don't force it — sometimes a supportive or curious 
response is better.

Post title: {post_title}
Post by: {post_author}
Post content: {post_content}

Write ONLY the comment text, nothing else. Keep it under 150 words."""

    comment = generate(cfg, prompt)
    if not comment:
        return

    metadata = {
        "post_id": post_id,
        "post_title": post_title,
        "post_author": post_author,
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
            cfg["llama_server_url"] = server_url
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

    while True:
        print("  ┌─────────────────────────────────────┐")
        print("  │  [1] Draft a new post                │")
        print("  │  [2] Browse feed & draft comments    │")
        print("  │  [3] Search & engage                 │")
        print("  │  [4] Review pending drafts            │")
        print("  │  [5] Check feed                       │")
        print("  │  [6] View profile                     │")
        print("  │  [7] Create dharma submolt            │")
        print("  │  [8] Settings                         │")
        print("  │  [q] Quit                             │")
        print("  └─────────────────────────────────────┘")

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
            cfg["llama_server_url"] = url
            save_config(cfg)
            print(f"  ✅ llama-server URL set to {url}")
    elif choice == "s":
        sub = input(f"  Default submolt (current: {cfg.get('default_submolt', 'general')}): ").strip()
        if sub:
            cfg["default_submolt"] = sub
            save_config(cfg)
            print(f"  ✅ Default submolt set to {sub}")


if __name__ == "__main__":
    main()