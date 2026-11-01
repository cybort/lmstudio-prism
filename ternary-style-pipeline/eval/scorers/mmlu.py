"""MMLU letter accuracy (Hendrycks et al.)."""

from __future__ import annotations

import re
from typing import Any

_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_ANSWER_RE = re.compile(
    r"(?:answer|答案|choice|选项)\s*(?:is|=|:|：)?\s*([ABCD])\b",
    re.IGNORECASE,
)


def extract_letter(text: str) -> str | None:
    if not text:
        return None
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    # Prefer last standalone capital letter A-D (lm-eval style heuristic)
    hits = _LETTER_RE.findall(text)
    if hits:
        return hits[-1].upper()
    return None


def score_mmlu_letter(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Each row: {id, gold: 'A'|'B'|'C'|'D', output}."""
    correct = 0
    details = []
    for r in rows:
        gold = str(r["gold"]).strip().upper()
        pred = extract_letter(str(r.get("output", "")))
        ok = pred == gold
        correct += int(ok)
        details.append({"id": r.get("id"), "gold": gold, "pred": pred, "pass": ok})
    n = len(rows)
    return {
        "metric": "accuracy",
        "accuracy": (correct / n) if n else 0.0,
        "n": n,
        "correct": correct,
        "protocol": "mmlu_letter_accuracy",
        "details": details,
    }
