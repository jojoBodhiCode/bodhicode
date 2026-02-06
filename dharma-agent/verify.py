"""
Post-generation verification for the Dharma Scholar agent.

Extracts claimed text names and teacher references from generated content,
checks them against the known-entities database, and produces a confidence report.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from entities import EntityDatabase


@dataclass
class EntityMention:
    """A mention of a Buddhist entity found in generated text."""
    name: str
    entity_type: str       # "text", "teacher", "school", "unknown"
    confidence: float      # 0.0 to 1.0
    context: str = ""      # surrounding text for review


@dataclass
class VerificationReport:
    """Results of verifying a piece of generated content."""
    verified: List[EntityMention] = field(default_factory=list)
    unverified: List[EntityMention] = field(default_factory=list)
    overall_confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return len(self.unverified) > 0 or len(self.warnings) > 0


# ─── Entity extraction patterns ─────────────────────────────────────────────

# Pattern for sutta/sutra references like "MN 26", "DN 2", "SN 12.2", "AN 3.65"
SUTTA_REF_PATTERN = re.compile(
    r'\b(MN|DN|SN|AN|Dhp|Ud|Iti|Snp|Thag|Thig|Kp|Vv|Pv)\s*(\d+(?:\.\d+)*)\b'
)

# Pattern for Tohoku numbers like "Toh 123", "Toh. 456"
TOH_PATTERN = re.compile(r'\bToh\.?\s*(\d+)\b', re.IGNORECASE)

# Pattern for quoted text names (in italics-like or quoted contexts)
# Matches "the X Sutra", "the X Sutta", "the X Tantra", etc.
TEXT_NAME_PATTERN = re.compile(
    r'\b(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+'
    r'(Sutra|Sutta|Tantra|Shastra|Karika|Stotra|Vinaya|Abhidharma)\b'
)

# Pattern for "X says", "X teaches", "X argues", "according to X"
TEACHER_REF_PATTERN = re.compile(
    r'(?:according to|as\s+(?:taught|explained|argued|stated)\s+by|'
    r'(?:taught|explains?|argues?|wrote|composed|said)\s+(?:by|that)\s*)'
    r'\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    re.IGNORECASE
)

# Pattern for possessive teacher references: "Nagarjuna's", "Tsongkhapa's"
POSSESSIVE_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+"
    r"(?:teaching|argument|philosophy|view|analysis|work|text|commentary|"
    r"Mulamadhyamakakarika|Madhyamakavatara|Bodhicaryavatara|"
    r"Prasannapada|Abhidharmakosa|Pramanavarttika)"
)

# Pattern for "in the X" where X is a text name
IN_THE_PATTERN = re.compile(
    r'\bin\s+the\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Sutra|Sutta|Tantra|Nikaya|Canon|'
    r'Pitaka|literature|tradition))?)(?:\s|,|\.|;)',
    re.IGNORECASE
)

# Pattern for verse/chapter references that might be fabricated
VERSE_REF_PATTERN = re.compile(
    r'\b(?:verse|chapter|section|passage|stanza)\s+(\d+(?:[.:]\d+)*)\b',
    re.IGNORECASE
)


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """
    Extract potential Buddhist entity mentions from generated text.

    Returns list of (entity_name, extraction_context) tuples.
    """
    mentions = []
    seen = set()

    # Sutta references (MN 26, etc.)
    for match in SUTTA_REF_PATTERN.finditer(text):
        name = f"{match.group(1)} {match.group(2)}"
        if name not in seen:
            ctx = text[max(0, match.start()-30):match.end()+30]
            mentions.append((name, ctx))
            seen.add(name)

    # Tohoku references
    for match in TOH_PATTERN.finditer(text):
        name = f"Toh {match.group(1)}"
        if name not in seen:
            ctx = text[max(0, match.start()-30):match.end()+30]
            mentions.append((name, ctx))
            seen.add(name)

    # Named texts (e.g., "Heart Sutra", "Diamond Sutra")
    for match in TEXT_NAME_PATTERN.finditer(text):
        name = f"{match.group(1)} {match.group(2)}"
        if name not in seen:
            ctx = text[max(0, match.start()-30):match.end()+30]
            mentions.append((name, ctx))
            seen.add(name)

    # Teacher references
    for match in TEACHER_REF_PATTERN.finditer(text):
        name = match.group(1).strip()
        if name not in seen and len(name) > 2:
            ctx = text[max(0, match.start()-20):match.end()+20]
            mentions.append((name, ctx))
            seen.add(name)

    # Possessive teacher references
    for match in POSSESSIVE_PATTERN.finditer(text):
        name = match.group(1).strip()
        if name not in seen and len(name) > 2:
            ctx = text[max(0, match.start()-20):match.end()+40]
            mentions.append((name, ctx))
            seen.add(name)

    # "In the X" references
    for match in IN_THE_PATTERN.finditer(text):
        name = match.group(1).strip()
        if name not in seen and len(name) > 3:
            ctx = text[max(0, match.start()-20):match.end()+20]
            mentions.append((name, ctx))
            seen.add(name)

    return mentions


def verify_content(text: str, entity_db: Optional[EntityDatabase] = None) -> VerificationReport:
    """
    Verify a piece of generated content against the known-entities database.

    Args:
        text: The generated text to verify
        entity_db: EntityDatabase instance (creates default if None)

    Returns:
        VerificationReport with verified/unverified entities and confidence score
    """
    if entity_db is None:
        entity_db = EntityDatabase()

    report = VerificationReport()

    # Extract entity mentions
    mentions = extract_entities(text)

    # Verify each mention
    for name, ctx in mentions:
        entity_type, confidence = entity_db.verify_entity(name)

        mention = EntityMention(
            name=name,
            entity_type=entity_type,
            confidence=confidence,
            context=ctx.strip(),
        )

        if confidence >= 0.5:
            report.verified.append(mention)
        else:
            report.unverified.append(mention)

    # Check for specific warning patterns
    # 1. Direct quotes that might be fabricated
    quote_pattern = re.compile(r'"[^"]{20,}"')  # Quotes longer than 20 chars
    quotes = quote_pattern.findall(text)
    if quotes:
        report.warnings.append(
            f"Contains {len(quotes)} direct quote(s) — verify these are not fabricated"
        )

    # 2. Specific verse numbers (high risk of fabrication)
    verse_refs = VERSE_REF_PATTERN.findall(text)
    if verse_refs:
        report.warnings.append(
            f"Contains specific verse/chapter references ({', '.join(verse_refs[:3])}) — verify accuracy"
        )

    # 3. Check for the pejorative "Hinayana"
    if re.search(r'\bHinayana\b', text, re.IGNORECASE):
        report.warnings.append(
            "'Hinayana' is widely considered pejorative — use 'Theravada' or 'early Buddhist schools'"
        )

    # Calculate overall confidence
    total = len(report.verified) + len(report.unverified)
    if total > 0:
        verified_score = sum(m.confidence for m in report.verified)
        total_possible = total  # Max 1.0 per entity
        report.overall_confidence = verified_score / total_possible
    else:
        report.overall_confidence = 0.8

    # Warnings reduce confidence
    report.overall_confidence -= 0.05 * len(report.warnings)
    report.overall_confidence = max(0.0, min(1.0, report.overall_confidence))

    return report


def format_verification_report(report: VerificationReport) -> str:
    """Format a verification report for display in the draft review UI."""
    lines = []

    if report.verified:
        lines.append("  [OK] Verified entities:")
        for m in report.verified:
            lines.append(f"       {m.name} ({m.entity_type}, {m.confidence:.0%})")

    if report.unverified:
        lines.append("  [??] Unverified entities (possible fabrication):")
        for m in report.unverified:
            lines.append(f"       {m.name} -- not found in known entities database")

    if report.warnings:
        lines.append("  [!!] Warnings:")
        for w in report.warnings:
            lines.append(f"       {w}")

    confidence_bar = "█" * int(report.overall_confidence * 10) + "░" * (10 - int(report.overall_confidence * 10))
    lines.append(f"  Confidence: [{confidence_bar}] {report.overall_confidence:.0%}")

    return "\n".join(lines)
