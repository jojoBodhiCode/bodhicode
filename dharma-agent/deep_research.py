"""
Deep Research Mode for the Dharma Scholar agent.

Autonomous research loop: given a high-level goal, the agent plans research
steps, queries the knowledge base, writes markdown notes, updates its journal,
and synthesizes a final document. Generated notes are indexed back into
ChromaDB for future reference.

Works with both:
  - The Discord bot (async, via generate_with_lock)
  - The CLI agent (sync, via generate_chat wrapper)

The key abstraction is `llm_func(messages, max_tokens=1024)` — a callable
that handles the actual LLM HTTP call. The caller provides the appropriate
version (async or sync).
"""

import re
from datetime import datetime, timezone
from pathlib import Path

from prompts import SYSTEM_PROMPT, TEMPERATURE_CREATIVE, TEMPERATURE_FACTUAL
from journal import add_entry as journal_add, format_for_prompt as journal_prompt

# Project output lives under the config dir
CONFIG_DIR = Path.home() / ".config" / "dharma-agent"
PROJECTS_DIR = CONFIG_DIR / "projects"


def slugify(text, max_len=60):
    """Convert text to a filesystem-safe slug."""
    slug = re.sub(r'[^\w\s-]', '', text.lower()).strip()
    slug = re.sub(r'[\s_-]+', '-', slug)
    return slug[:max_len].rstrip('-')


class DeepResearch:
    """Tracks state for a deep research session."""

    def __init__(self, goal, max_steps=8, project_dir=None):
        self.goal = goal
        self.max_steps = max_steps
        slug = slugify(goal)
        self.project_dir = project_dir or (PROJECTS_DIR / slug)
        self.notes_dir = self.project_dir / "notes"
        self.steps = []            # [(title, description), ...]
        self.progress_summary = ""
        self.current_step = 0
        self.status = "pending"    # pending | planning | researching | synthesizing | done | stopped
        self.note_files = []       # list of Path objects for generated MD files

    def setup_dirs(self):
        """Create project and notes directories."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)


# ─── Research prompts ────────────────────────────────────────────────────────

PLAN_PROMPT = """\
You are a research planner. Given the following goal, break it into {max_steps} \
concrete research steps. Each step should focus on a specific subtopic that \
contributes to the overall goal.

GOAL: {goal}

{rag_context}

Output a numbered list in this exact format (one step per line):
1. Step Title: Brief description of what to research
2. Step Title: Brief description of what to research
...

Output ONLY the numbered list, nothing else."""

STEP_PROMPT = """\
You are in deep research mode.

GOAL: {goal}

STEP {step_num} of {total_steps}: {step_title}
{step_description}

PROGRESS SO FAR:
{progress_summary}

{rag_section}

Write detailed research notes for this step in markdown format. Use a heading \
with the step title, then well-structured content with subheadings as needed. \
Include specific references to source texts when available. \
End with a "## Key Findings" section containing a brief bullet list of the \
most important points from this step."""

SUMMARY_PROMPT = """\
You just completed a research step. Here are the notes you wrote:

{notes}

Write a brief progress summary (3-5 sentences) capturing the key findings \
from ALL research so far, including this step. Previous progress:
{previous_progress}

Output ONLY the summary paragraph, nothing else."""

SYNTHESIS_PROMPT = """\
You have completed a deep research project. Below are all the research notes \
from each step. Synthesize them into a single cohesive document with a clear \
structure, introduction, and conclusion.

GOAL: {goal}

{all_notes}

Write a well-structured final document in markdown format. Include:
- A title and introduction explaining the goal
- Organized sections drawing from the research
- Cross-references between related findings
- A conclusion summarizing the key insights
- A "Sources Referenced" section at the end"""


# ─── Core research functions ─────────────────────────────────────────────────

def _build_rag_context(query, rag, k=5):
    """Query RAG and return formatted context string + source IDs."""
    if rag is None:
        return "", []
    try:
        if rag.collection.count() == 0:
            return "", []
        context, sources = rag.retrieve(query, k=k)
        source_ids = []
        for s in sources:
            sid = s.get("text_id", s.get("source", ""))
            if sid and sid not in source_ids:
                source_ids.append(sid)
        return context, source_ids
    except Exception:
        return "", []


def _parse_plan(text):
    """Parse a numbered list into [(title, description), ...] tuples."""
    steps = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Match: "1. Title: Description" or "1. Title - Description"
        m = re.match(r'^\d+[\.\)]\s*(.+?)(?:\s*[:–—-]\s*)(.+)$', line)
        if m:
            steps.append((m.group(1).strip(), m.group(2).strip()))
        elif re.match(r'^\d+[\.\)]\s*', line):
            # Just a numbered line without clear title:description split
            content = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            steps.append((content[:60], content))
    return steps


def plan_research(session, rag, llm_func):
    """
    Create a research plan by asking the LLM to break the goal into steps.

    Args:
        session: DeepResearch instance
        rag: DharmaRAG instance (or None)
        llm_func: callable(messages, max_tokens) -> str

    Returns:
        List of (title, description) tuples. Also saves plan.md.
    """
    session.status = "planning"

    # Get RAG context for the overall goal
    rag_context, source_ids = _build_rag_context(session.goal, rag)
    rag_block = ""
    if rag_context:
        rag_block = (
            "The following source material may help inform your plan:\n\n"
            f"{rag_context}"
        )

    prompt = PLAN_PROMPT.format(
        goal=session.goal,
        max_steps=session.max_steps,
        rag_context=rag_block,
    )

    # Include journal memory for continuity
    journal_context = journal_prompt()
    system = SYSTEM_PROMPT
    if journal_context:
        system += "\n\n" + journal_context

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    response = llm_func(messages, max_tokens=1024)
    if not response:
        # Fallback: single generic step
        session.steps = [("Research the topic", session.goal)]
        return session.steps

    steps = _parse_plan(response)
    if not steps:
        # Fallback: treat each line as a step
        for line in response.strip().splitlines():
            line = line.strip()
            if line:
                steps.append((line[:60], line))

    # Cap at max_steps
    steps = steps[:session.max_steps]
    session.steps = steps

    # Save plan.md
    plan_md = f"# Research Plan\n\n**Goal:** {session.goal}\n\n"
    plan_md += f"**Created:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
    plan_md += f"**Steps:** {len(steps)}\n\n"
    for i, (title, desc) in enumerate(steps, 1):
        plan_md += f"{i}. **{title}**: {desc}\n"
    if source_ids:
        plan_md += f"\n**Sources consulted during planning:** {', '.join(source_ids)}\n"

    plan_path = session.project_dir / "plan.md"
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan_md)
    session.note_files.append(plan_path)

    # Journal entry
    journal_add(
        user="DharmaScholar",
        channel="deep-research",
        question=f"Started deep research: {session.goal[:100]}",
        sources=source_ids,
        response_snippet=f"Plan: {len(steps)} steps",
    )

    return steps


def execute_step(session, step_num, rag, llm_func):
    """
    Execute a single research step: RAG query, LLM generation, save notes.

    Args:
        session: DeepResearch instance
        step_num: 0-based step index
        rag: DharmaRAG instance (or None)
        llm_func: callable(messages, max_tokens) -> str

    Returns:
        The generated notes text, or None on failure.
    """
    session.status = "researching"
    session.current_step = step_num

    title, description = session.steps[step_num]

    # Query RAG for this specific subtopic
    rag_query = f"{session.goal} - {title}: {description}"
    rag_context, source_ids = _build_rag_context(rag_query, rag)

    rag_section = ""
    if rag_context:
        rag_section = (
            "The following canonical texts are relevant to this step. "
            "Ground your research in these sources and cite them.\n\n"
            f"{rag_context}"
        )

    prompt = STEP_PROMPT.format(
        goal=session.goal,
        step_num=step_num + 1,
        total_steps=len(session.steps),
        step_title=title,
        step_description=description,
        progress_summary=session.progress_summary or "(This is the first step.)",
        rag_section=rag_section,
    )

    journal_context = journal_prompt()
    system = SYSTEM_PROMPT
    if journal_context:
        system += "\n\n" + journal_context

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    notes = llm_func(messages, max_tokens=2048)
    if not notes:
        return None

    # Save notes as MD file
    step_slug = slugify(title, max_len=40)
    filename = f"{step_num + 1:02d}_{step_slug}.md"
    note_path = session.notes_dir / filename

    note_content = notes
    if source_ids:
        note_content += f"\n\n---\n*Sources: {', '.join(source_ids)}*\n"

    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note_content)
    session.note_files.append(note_path)

    # Journal entry
    journal_add(
        user="DharmaScholar",
        channel="deep-research",
        question=f"Step {step_num + 1}/{len(session.steps)}: {title}",
        sources=source_ids,
        response_snippet=notes[:150],
    )

    # Update progress summary
    summary_messages = [
        {"role": "system", "content": "You are a concise research assistant."},
        {"role": "user", "content": SUMMARY_PROMPT.format(
            notes=notes[:1500],  # truncate to fit context
            previous_progress=session.progress_summary or "(No previous progress.)",
        )},
    ]
    summary = llm_func(summary_messages, max_tokens=512)
    if summary:
        session.progress_summary = summary.strip()

    return notes


def synthesize_research(session, llm_func):
    """
    Synthesize all research notes into a final cohesive document.

    Args:
        session: DeepResearch instance
        llm_func: callable(messages, max_tokens) -> str

    Returns:
        The synthesis text, or None on failure.
    """
    session.status = "synthesizing"

    # Read all note files from the notes directory
    all_notes = ""
    for note_file in sorted(session.notes_dir.glob("*.md")):
        with open(note_file, "r", encoding="utf-8") as f:
            content = f.read()
        all_notes += f"\n\n{'=' * 60}\n{note_file.name}\n{'=' * 60}\n\n{content}"

    if not all_notes.strip():
        return None

    # Truncate if too long for context
    max_notes_chars = 6000
    if len(all_notes) > max_notes_chars:
        # Use progress summary + truncated notes
        all_notes = (
            f"ACCUMULATED SUMMARY:\n{session.progress_summary}\n\n"
            f"DETAILED NOTES (truncated):\n{all_notes[:max_notes_chars]}"
        )

    prompt = SYNTHESIS_PROMPT.format(
        goal=session.goal,
        all_notes=all_notes,
    )

    journal_context = journal_prompt()
    system = SYSTEM_PROMPT
    if journal_context:
        system += "\n\n" + journal_context

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    synthesis = llm_func(messages, max_tokens=4096)
    if not synthesis:
        return None

    # Save final.md
    final_path = session.project_dir / "final.md"
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(synthesis)
    session.note_files.append(final_path)

    # Journal entry
    journal_add(
        user="DharmaScholar",
        channel="deep-research",
        question=f"Completed deep research: {session.goal[:100]}",
        sources=[],
        response_snippet=synthesis[:150],
    )

    session.status = "done"
    return synthesis


def index_research_notes(session, rag):
    """
    Index generated research notes back into ChromaDB.

    Args:
        session: DeepResearch instance
        rag: DharmaRAG instance (or None)

    Returns:
        Number of chunks indexed.
    """
    if rag is None:
        return 0

    try:
        from ingest.ingest_common import chunk_text, enrich_metadata
    except ImportError:
        return 0

    slug = session.project_dir.name
    total_indexed = 0

    # Detect tradition from the goal text
    tradition = _detect_tradition(session.goal)

    for md_file in session.project_dir.rglob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            continue

        text_id = f"DR:{slug}/{md_file.name}"
        chunks = chunk_text(content, base_metadata={})
        chunks = enrich_metadata(
            chunks,
            tradition=tradition or "General",
            text_id=text_id,
            text_type="research_note",
            source_url="",
        )

        count = rag.index_chunks(chunks)
        total_indexed += count

    return total_indexed


def _detect_tradition(text):
    """Simple tradition detection from text keywords."""
    text_lower = text.lower()
    keywords = {
        "Theravada": [
            "theravada", "pali", "sutta", "nikaya", "vipassana",
            "dukkha", "anatta", "dhammapada",
        ],
        "Mahayana": [
            "mahayana", "madhyamaka", "yogacara", "nagarjuna",
            "sunyata", "bodhisattva", "prajnaparamita", "zen",
        ],
        "Vajrayana": [
            "vajrayana", "tibetan", "tantra", "dzogchen", "mahamudra",
            "kalachakra", "mandala", "tsongkhapa",
        ],
    }
    best = None
    best_score = 0
    for tradition, kws in keywords.items():
        score = sum(1 for kw in kws if kw in text_lower)
        if score > best_score:
            best_score = score
            best = tradition
    return best
