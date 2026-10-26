"""Four-stage HR policy agent: retrieve, check, answer, or escalate."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hr_agent.retrieve import ScoredSection, _tokenize, search

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TICKET_PATH = DATA_DIR / "escalations.jsonl"

_MIN_SCORE = 4
_GENERIC = {
    "apply",
    "available",
    "benefit",
    "company",
    "day",
    "document",
    "eligible",
    "employee",
    "leave",
    "medical",
    "paid",
    "policy",
    "process",
    "request",
    "required",
    "service",
    "work",
    "working",
}
_SENSITIVE = re.compile(
    r"\b(unfair|harass\w*|discriminat\w*|lawsuit|grievance|complaint|exception|waive|disciplinary|dispute)\b"
    r"|\bmy manager\b|\bapprove my\b|\bbend the rules?\b|\bfire me\b|\bterminate me\b"
    r"|\bmedical exception\b|\bemployee id\b|\bstaff id\b|\blegal action\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

_REASONS = {
    "sensitive_or_personal": (
        "This question is sensitive, personal, or asks for an exception. "
        "It has been sent to an HR specialist, and no policy answer was guessed."
    ),
    "unclear": (
        "This question is unclear, so a single policy section cannot be chosen safely. "
        "It has been sent to an HR specialist."
    ),
    "outside_policy": (
        "The handbook does not cover this question. It has been sent to an HR specialist."
    ),
    "model_unsupported": (
        "A grounded answer could not be prepared from the retrieved policy text. "
        "The question has been sent to an HR specialist."
    ),
}


@dataclass(frozen=True)
class AgentResult:
    action: str
    message: str
    citations: tuple[str, ...]
    reason: str | None
    ticket_id: str | None
    stages: tuple[dict[str, str], ...]
    used_model: bool
    input_tokens: int
    output_tokens: int
    elapsed_ms: int


def answer_question(question: str) -> AgentResult:
    started = time.perf_counter()
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be empty")

    _load_env()
    matches = search(cleaned)
    retrieve_detail = _retrieve_detail(matches)
    sensitive = _SENSITIVE.search(cleaned) is not None
    covered = _is_covered(cleaned, matches)

    if sensitive or not covered:
        reason = "sensitive_or_personal" if sensitive else _coverage_reason(cleaned)
        ticket_id = _write_ticket(cleaned, reason)
        elapsed = int((time.perf_counter() - started) * 1000)
        check_detail = _REASONS[reason]
        return AgentResult(
            action="escalate",
            message=f"{check_detail} Ticket {ticket_id}.",
            citations=(),
            reason=reason,
            ticket_id=ticket_id,
            stages=(
                {"name": "Retrieve", "detail": retrieve_detail},
                {"name": "Check", "detail": check_detail},
                {"name": "Escalate", "detail": f"Saved ticket {ticket_id}."},
            ),
            used_model=False,
            input_tokens=0,
            output_tokens=0,
            elapsed_ms=elapsed,
        )

    section = matches[0].section
    citation = f"{section.citation} ({section.source})"
    quote = f"Based on {section.citation}:\n\n{section.text}"
    model_text, input_tokens, output_tokens, model_note = _phrase_with_model(cleaned, section.citation, section.text)
    used_model = False
    message = quote
    answer_detail = "No model key is set, so the answer is the handbook section itself."

    if model_text == "ESCALATE":
        ticket_id = _write_ticket(cleaned, "model_unsupported")
        elapsed = int((time.perf_counter() - started) * 1000)
        check_detail = "The question matched a policy section, but a supported answer could not be confirmed."
        return AgentResult(
            action="escalate",
            message=f"{_REASONS['model_unsupported']} Ticket {ticket_id}.",
            citations=(citation,),
            reason="model_unsupported",
            ticket_id=ticket_id,
            stages=(
                {"name": "Retrieve", "detail": retrieve_detail},
                {"name": "Check", "detail": check_detail},
                {"name": "Escalate", "detail": f"Saved ticket {ticket_id}."},
            ),
            used_model=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=elapsed,
        )

    if model_text:
        used_model = True
        message = model_text
        answer_detail = model_note
    elif model_note:
        answer_detail = model_note

    elapsed = int((time.perf_counter() - started) * 1000)
    return AgentResult(
        action="answer",
        message=message,
        citations=(citation,),
        reason=None,
        ticket_id=None,
        stages=(
            {"name": "Retrieve", "detail": retrieve_detail},
            {
                "name": "Check",
                "detail": f"{section.citation} covers this question, so it can be answered from the handbook.",
            },
            {"name": "Answer", "detail": answer_detail},
        ),
        used_model=used_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_ms=elapsed,
    )


def _is_covered(question: str, matches: list[ScoredSection]) -> bool:
    if not matches or matches[0].score < _MIN_SCORE:
        return False
    specific_query = _specific_terms(question)
    if not specific_query:
        return False
    return specific_query <= set(matches[0].overlap)


def _coverage_reason(question: str) -> str:
    if len(_specific_terms(question)) <= 1:
        return "unclear"
    return "outside_policy"


def _specific_terms(question: str) -> set[str]:
    return {token for token in _tokenize(question) if token not in _GENERIC}


def _retrieve_detail(matches: list[ScoredSection]) -> str:
    if not matches:
        return "No handbook section shared words with the question."
    parts = [
        f"{item.section.citation} (score {item.score})"
        for item in matches
    ]
    return "Top sections: " + "; ".join(parts) + "."


def _phrase_with_model(question: str, citation: str, passage: str) -> tuple[str | None, int, int, str]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, 0, 0, ""

    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    prompt = (
        "You answer employee HR questions using only the policy passage.\n"
        "If the passage does not contain the answer, reply with exactly ESCALATE.\n"
        "Do not add rules, numbers, or exceptions that are not in the passage.\n"
        "Write a short answer and mention the section name.\n\n"
        f"Section: {citation}\n{passage}\n\nQuestion: {question}"
    )
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You only use the policy passage provided by the user."},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None, _estimate_tokens(prompt), 0, "The model call failed, so the handbook section is shown instead."

    content = str(body.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    usage = body.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or _estimate_tokens(prompt))
    output_tokens = int(usage.get("completion_tokens") or _estimate_tokens(content))
    if not content:
        return None, input_tokens, output_tokens, "The model returned an empty answer, so the handbook section is shown instead."
    if content.upper().startswith("ESCALATE"):
        return "ESCALATE", input_tokens, output_tokens, ""
    extra_numbers = _NUMBER.findall(content)
    allowed_numbers = set(_NUMBER.findall(passage))
    if any(number not in allowed_numbers for number in extra_numbers):
        return None, input_tokens, output_tokens, "The model draft added figures that are not in the policy, so the handbook text is shown instead."
    if not content.lower().startswith("based on"):
        content = f"Based on {citation}:\n\n{content}"
    return content, input_tokens, output_tokens, "The answer was phrased from the retrieved section only."


def _write_ticket(question: str, reason: str) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ticket_id = uuid.uuid4().hex[:12]
    record = {
        "id": ticket_id,
        "time": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "reason": reason,
    }
    with TICKET_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return ticket_id


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _load_env() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
