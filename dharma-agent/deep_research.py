"""
Deep Research Mode for the Dharma Scholar agent.

Autonomous research loop: given a high-level goal, the agent plans research
steps, queries the knowledge base, writes markdown notes, updates its journal,
and synthesizes a final document. Generated notes are indexed back into
ChromaDB for future reference.

Features:
  - Dual RAG queries per step (broad + focused, no extra LLM cost)
  - Wikipedia fallback when RAG returns thin results (no LLM cost)
  - Self-critique folded into the summary step (0 extra LLM calls)
  - Dynamic plan revision from summary feedback (0 extra LLM calls)
  - Iterative deepening: gap analysis after synthesis + one deepening pass
  - Resume interrupted sessions from existing project directories

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

# Wikipedia fallback (lazy-loaded, graceful if unavailable)
_wiki_available = None

def _get_wiki_lookup():
    """Lazy-import wiki_lookup module. Returns the lookup function or None."""
    global _wiki_available
    if _wiki_available is not None:
        return _wiki_available
    try:
        from wiki_lookup import lookup as wiki_lookup
        _wiki_available = wiki_lookup
        return wiki_lookup
    except ImportError:
        _wiki_available = False
        return None

# Minimum RAG source count before we consider Wikipedia fallback
RAG_THIN_THRESHOLD = 2

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
        self.status = "pending"    # pending | planning | researching | synthesizing | deepening | done | stopped
        self.note_files = []       # list of Path objects for generated MD files
        self.gaps = []             # identified gaps for deepening pass
        self.iteration = 0         # 0 = initial pass, 1 = deepening pass
        self.critique = ""         # rolling self-critique notes

    def setup_dirs(self):
        """Create project and notes directories."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def resume(cls, project_dir):
        """
        Resume an interrupted session from an existing project directory.

        Reads plan.md to reconstruct the goal and steps, then determines
        which steps have already been completed based on existing note files.

        Returns a DeepResearch instance ready to continue, or None if the
        directory doesn't contain a valid session.
        """
        project_dir = Path(project_dir)
        plan_path = project_dir / "plan.md"
        if not plan_path.exists():
            return None

        with open(plan_path, "r", encoding="utf-8") as f:
            plan_text = f.read()

        # Extract goal from plan.md
        goal_match = re.search(r'\*\*Goal:\*\*\s*(.+)', plan_text)
        if not goal_match:
            return None
        goal = goal_match.group(1).strip()

        # Extract steps from plan.md
        steps = []
        for line in plan_text.splitlines():
            m = re.match(r'^\d+\.\s+\*\*(.+?)\*\*:\s*(.+)$', line)
            if m:
                steps.append((m.group(1).strip(), m.group(2).strip()))

        if not steps:
            return None

        session = cls(goal, max_steps=len(steps), project_dir=project_dir)
        session.steps = steps
        session.note_files = [plan_path]

        # Determine which steps are already done by checking note files
        notes_dir = project_dir / "notes"
        completed = 0
        last_note_content = ""
        if notes_dir.exists():
            for note_file in sorted(notes_dir.glob("*.md")):
                session.note_files.append(note_file)
                completed += 1
                with open(note_file, "r", encoding="utf-8") as f:
                    last_note_content = f.read()

        session.current_step = completed

        # Rebuild a rough progress summary from the last note
        if last_note_content:
            # Use the key findings section if present, otherwise first 500 chars
            findings_match = re.search(
                r'##\s*Key Findings(.+?)(?=\n##|\Z)', last_note_content, re.DOTALL
            )
            if findings_match:
                session.progress_summary = findings_match.group(1).strip()[:500]
            else:
                session.progress_summary = last_note_content[:500]

        # Check if final.md already exists (session was fully complete)
        final_path = project_dir / "final.md"
        if final_path.exists():
            session.status = "done"
            session.note_files.append(final_path)
        else:
            session.status = "researching"

        return session


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

{critique_section}

{rag_section}

Write detailed research notes for this step in markdown format. Use a heading \
with the step title, then well-structured content with subheadings as needed. \
Include specific references to source texts when available. \
End with a "## Key Findings" section containing a brief bullet list of the \
most important points from this step."""

# The summary prompt now also extracts self-critique and plan suggestions,
# all in one LLM call (no extra compute).
SUMMARY_PROMPT = """\
You just completed a research step. Here are the notes you wrote:

{notes}

Previous progress: {previous_progress}

Respond in EXACTLY this format (keep each section to 2-3 sentences max):

SUMMARY: Brief progress summary capturing key findings from all research so far.

GAPS: Any weak areas, unsupported claims, or topics that need deeper investigation. Write "none" if the research is solid.

PLAN: If a new research step should be added to better serve the goal, suggest it here. Write "none" if the current plan is sufficient."""

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

GAP_ANALYSIS_PROMPT = """\
You have completed initial research on this goal:

GOAL: {goal}

Here is the accumulated progress summary:
{progress_summary}

Here are the noted gaps and weak areas from the research:
{critique}

Identify the 2-3 most important gaps that would significantly improve the \
final output if addressed. For each gap, write a research step in this format:

1. Step Title: Brief description of what to research

Output ONLY the numbered list. If the research is already thorough, output \
"NO GAPS" and nothing else."""


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


def _build_wiki_context(query, max_articles=3):
    """
    Fetch Wikipedia summaries as fallback context.

    Returns (context_str, source_urls) or ("", []) if unavailable.
    No LLM cost — just Wikipedia API calls.
    """
    wiki_lookup = _get_wiki_lookup()
    if not wiki_lookup:
        return "", []
    try:
        return wiki_lookup(query, max_articles=max_articles)
    except Exception:
        return "", []


def _build_dual_rag_context(goal, step_title, step_description, rag, k=4):
    """
    Two RAG queries per step for broader coverage, plus Wikipedia fallback.

    Query 1: Focused on the specific step topic
    Query 2: Step topic in the context of the overall goal
    Fallback: Wikipedia lookup if RAG returns fewer than RAG_THIN_THRESHOLD sources

    Results are merged and deduplicated by source ID.
    """
    # Focused query on step topic
    context1, ids1 = _build_rag_context(f"{step_title}: {step_description}", rag, k=k)

    # Broader query connecting step to goal
    context2, ids2 = _build_rag_context(f"{goal} — {step_title}", rag, k=3)

    # Merge, dedup by source ID
    all_ids = list(ids1)
    for sid in ids2:
        if sid not in all_ids:
            all_ids.append(sid)

    # Combine context (avoid duplicating if both returned the same chunks)
    if context2 and context2 != context1:
        combined = f"{context1}\n\n{context2}"
    else:
        combined = context1

    # Wikipedia fallback when RAG is thin
    wiki_context = ""
    wiki_urls = []
    if len(all_ids) < RAG_THIN_THRESHOLD:
        wiki_query = f"{step_title} {step_description} Buddhism"
        wiki_context, wiki_urls = _build_wiki_context(wiki_query, max_articles=2)

    if wiki_context:
        if combined:
            combined += (
                "\n\n--- Wikipedia supplementary context ---\n\n"
                f"{wiki_context}"
            )
        else:
            combined = wiki_context
        # Add wiki URLs as source IDs with WP: prefix
        for url in wiki_urls:
            wp_id = f"WP:{url.split('/wiki/')[-1]}" if "/wiki/" in url else url
            if wp_id not in all_ids:
                all_ids.append(wp_id)

    return combined, all_ids


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


def _parse_summary_response(text):
    """
    Parse the structured summary response into (summary, gaps, plan_suggestion).

    Expected format:
        SUMMARY: ...
        GAPS: ...
        PLAN: ...
    """
    summary = ""
    gaps = ""
    plan_suggestion = ""

    current = None
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("SUMMARY:"):
            current = "summary"
            summary = stripped[len("SUMMARY:"):].strip()
        elif upper.startswith("GAPS:") or upper.startswith("GAP:"):
            current = "gaps"
            gaps = re.sub(r'^GAPS?:\s*', '', stripped, flags=re.IGNORECASE).strip()
        elif upper.startswith("PLAN:"):
            current = "plan"
            plan_suggestion = stripped[len("PLAN:"):].strip()
        elif current == "summary":
            summary += " " + stripped
        elif current == "gaps":
            gaps += " " + stripped
        elif current == "plan":
            plan_suggestion += " " + stripped

    # Treat "none" as empty
    if gaps.lower().strip() in ("none", "none.", "n/a", ""):
        gaps = ""
    if plan_suggestion.lower().strip() in ("none", "none.", "n/a", ""):
        plan_suggestion = ""

    return summary.strip(), gaps.strip(), plan_suggestion.strip()


def plan_research(session, rag, llm_func):
    """
    Create a research plan by asking the LLM to break the goal into steps.

    Returns:
        List of (title, description) tuples. Also saves plan.md.
    """
    session.status = "planning"

    # Get RAG context for the overall goal, with Wikipedia fallback
    rag_context, source_ids = _build_rag_context(session.goal, rag)
    if len(source_ids) < RAG_THIN_THRESHOLD:
        wiki_context, wiki_urls = _build_wiki_context(
            f"{session.goal} Buddhism", max_articles=2
        )
        if wiki_context:
            rag_context = f"{rag_context}\n\n{wiki_context}" if rag_context else wiki_context
            for url in wiki_urls:
                wp_id = f"WP:{url.split('/wiki/')[-1]}" if "/wiki/" in url else url
                if wp_id not in source_ids:
                    source_ids.append(wp_id)

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
    _save_plan_md(session, source_ids)

    # Journal entry
    journal_add(
        user="DharmaScholar",
        channel="deep-research",
        question=f"Started deep research: {session.goal[:100]}",
        sources=source_ids,
        response_snippet=f"Plan: {len(steps)} steps",
    )

    return steps


def _save_plan_md(session, source_ids=None):
    """Write or overwrite plan.md for the session."""
    plan_md = f"# Research Plan\n\n**Goal:** {session.goal}\n\n"
    plan_md += f"**Created:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
    iteration_label = " (deepening pass)" if session.iteration > 0 else ""
    plan_md += f"**Steps:** {len(session.steps)}{iteration_label}\n\n"
    for i, (title, desc) in enumerate(session.steps, 1):
        plan_md += f"{i}. **{title}**: {desc}\n"
    if source_ids:
        plan_md += f"\n**Sources consulted during planning:** {', '.join(source_ids)}\n"

    plan_path = session.project_dir / "plan.md"
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan_md)
    if plan_path not in session.note_files:
        session.note_files.append(plan_path)


def execute_step(session, step_num, rag, llm_func):
    """
    Execute a single research step: dual RAG query, LLM generation, save notes.

    The summary call also extracts self-critique and plan revision suggestions
    at zero extra LLM cost (folded into one prompt).

    Returns:
        The generated notes text, or None on failure.
    """
    session.status = "researching"
    session.current_step = step_num

    title, description = session.steps[step_num]

    # Dual RAG query: focused + broad (no LLM cost, just embeddings)
    rag_context, source_ids = _build_dual_rag_context(
        session.goal, title, description, rag
    )

    rag_section = ""
    if rag_context:
        rag_section = (
            "The following canonical texts are relevant to this step. "
            "Ground your research in these sources and cite them.\n\n"
            f"{rag_context}"
        )

    # Include accumulated critique so the agent addresses known gaps
    critique_section = ""
    if session.critique:
        critique_section = (
            "KNOWN GAPS FROM PRIOR STEPS (address these if relevant to this step):\n"
            f"{session.critique}"
        )

    prompt = STEP_PROMPT.format(
        goal=session.goal,
        step_num=step_num + 1,
        total_steps=len(session.steps),
        step_title=title,
        step_description=description,
        progress_summary=session.progress_summary or "(This is the first step.)",
        critique_section=critique_section,
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
    # Prefix with iteration for deepening pass notes
    prefix = f"d{session.iteration}_" if session.iteration > 0 else ""
    filename = f"{prefix}{step_num + 1:02d}_{step_slug}.md"
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

    # Combined summary + self-critique + plan suggestion (1 LLM call, not 3)
    summary_messages = [
        {"role": "system", "content": "You are a concise research assistant."},
        {"role": "user", "content": SUMMARY_PROMPT.format(
            notes=notes[:1500],  # truncate to fit context
            previous_progress=session.progress_summary or "(No previous progress.)",
        )},
    ]
    summary_response = llm_func(summary_messages, max_tokens=512)
    if summary_response:
        summary, gaps, plan_suggestion = _parse_summary_response(summary_response)
        if summary:
            session.progress_summary = summary
        if gaps:
            session.critique = gaps  # latest critique replaces previous

        # Dynamic plan revision: if the LLM suggests a new step, append it
        if plan_suggestion and len(session.steps) < session.max_steps + 3:
            new_steps = _parse_plan(f"1. {plan_suggestion}")
            if new_steps:
                session.steps.extend(new_steps)
                _save_plan_md(session)

    return notes


def synthesize_research(session, llm_func):
    """
    Synthesize all research notes into a final cohesive document.

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
    if final_path not in session.note_files:
        session.note_files.append(final_path)

    # Journal entry
    journal_add(
        user="DharmaScholar",
        channel="deep-research",
        question=f"Synthesized deep research: {session.goal[:100]}",
        sources=[],
        response_snippet=synthesis[:150],
    )

    return synthesis


def identify_gaps(session, llm_func):
    """
    After initial synthesis, identify gaps worth a deepening pass.

    Cost: 1 LLM call. Returns list of (title, description) for new steps,
    or empty list if research is already solid.
    """
    if not session.progress_summary:
        return []

    prompt = GAP_ANALYSIS_PROMPT.format(
        goal=session.goal,
        progress_summary=session.progress_summary,
        critique=session.critique or "(No specific gaps noted.)",
    )

    messages = [
        {"role": "system", "content": "You are a thorough research reviewer."},
        {"role": "user", "content": prompt},
    ]

    response = llm_func(messages, max_tokens=512)
    if not response:
        return []

    # Check for "NO GAPS" response
    if "NO GAPS" in response.upper():
        return []

    gaps = _parse_plan(response)
    return gaps[:3]  # cap at 3 deepening steps


def run_deepening_pass(session, rag, llm_func):
    """
    Execute a deepening pass: research identified gaps, then re-synthesize.

    Cost: 1 call per gap step (2-3) + 1 summary each + 1 synthesis = ~7-10 calls total.
    Capped at one deepening iteration.

    Returns:
        The new synthesis text, or None if no deepening was needed.
    """
    if session.iteration > 0:
        session.status = "done"
        return None  # only one deepening pass

    session.status = "deepening"
    session.iteration = 1

    gaps = identify_gaps(session, llm_func)
    if not gaps:
        session.status = "done"
        return None

    session.gaps = gaps
    # Replace the step list with gap steps for the deepening pass
    original_steps = session.steps
    session.steps = gaps

    journal_add(
        user="DharmaScholar",
        channel="deep-research",
        question=f"Deepening: {len(gaps)} gaps identified",
        sources=[],
        response_snippet="; ".join(t for t, _ in gaps),
    )

    # Research each gap
    for i in range(len(gaps)):
        if session.status == "stopped":
            break
        execute_step(session, i, rag, llm_func)

    # Restore full step list (original + deepening) for context
    session.steps = original_steps + gaps

    # Re-synthesize with all notes (original + deepening)
    synthesis = synthesize_research(session, llm_func)

    session.status = "done"
    return synthesis


def index_research_notes(session, rag):
    """
    Index generated research notes back into ChromaDB.

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
