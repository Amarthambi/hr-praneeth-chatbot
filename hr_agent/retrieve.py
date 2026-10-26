"""Find handbook sections that match an employee question."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

POLICIES_DIR = Path(__file__).resolve().parent.parent / "policies"

_STOPWORDS = {
    "a",
    "about",
    "am",
    "an",
    "and",
    "are",
    "be",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "many",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "should",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "would",
    "you",
    "your",
}

_ALIASES = {
    "application": "apply",
    "applications": "apply",
    "applied": "apply",
    "applying": "apply",
    "benefits": "benefit",
    "days": "day",
    "documentation": "document",
    "documented": "document",
    "documents": "document",
    "eligibility": "eligible",
    "employees": "employee",
}

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Section:
    source: str
    document: str
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.document} — {self.heading}"


@dataclass(frozen=True)
class ScoredSection:
    section: Section
    score: int
    overlap: tuple[str, ...]


def load_sections(policies_dir: Path | None = None) -> list[Section]:
    directory = policies_dir or POLICIES_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Policy folder not found: {directory}")

    sections: list[Section] = []
    for path in sorted(directory.glob("*.md")):
        document = ""
        heading = ""
        lines: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if raw_line.startswith("# "):
                _append_section(sections, path.name, document, heading, lines)
                document = raw_line[2:].strip()
                heading = ""
                lines = []
            elif raw_line.startswith("## "):
                _append_section(sections, path.name, document, heading, lines)
                heading = raw_line[3:].strip()
                lines = []
            else:
                lines.append(raw_line)
        _append_section(sections, path.name, document, heading, lines)
    return sections


def _append_section(
    sections: list[Section],
    filename: str,
    document: str,
    heading: str,
    lines: list[str],
) -> None:
    body = " ".join(line.strip() for line in lines if line.strip())
    if document and heading and body:
        sections.append(
            Section(
                source=f"policies/{filename}",
                document=document,
                heading=heading,
                text=body,
            )
        )


def search(question: str, limit: int = 3, policies_dir: Path | None = None) -> list[ScoredSection]:
    query_terms = set(_tokenize(question))
    if not query_terms or limit < 1:
        return []

    ranked: list[ScoredSection] = []
    for section in load_sections(policies_dir):
        body_tokens = _tokenize(section.text)
        heading_tokens = _tokenize(f"{section.heading} {section.document}")
        overlap = tuple(sorted(query_terms & (set(body_tokens) | set(heading_tokens))))
        if not overlap:
            continue
        heading_overlap = query_terms & set(heading_tokens)
        term_hits = sum(1 for token in body_tokens if token in query_terms)
        score = 2 * len(overlap) + 3 * len(heading_overlap) + min(term_hits, 8)
        ranked.append(ScoredSection(section=section, score=score, overlap=overlap))

    ranked.sort(key=lambda item: (-item.score, item.section.source, item.section.heading))
    return ranked[:limit]


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        if match in _STOPWORDS or len(match) < 2:
            continue
        tokens.append(_ALIASES.get(match, match))
    return tokens
