"""GSM8K flexible-extract accuracy (Cobbe et al. / Eleuther lm-eval protocol)."""

from __future__ import annotations

import re
from typing import Any


_ANS_RE = re.compile(
    r"(?:answer(?:\s*is)?|final(?:\s*answer)?|答案)\s*[:：]?\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_HASH_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_LAST_NUM_RE = re.compile(r"(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def normalize_number(s: str) -> str:
    s = s.replace(",", "").strip()
    if s.endswith(".0"):
        s = s[:-2]
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def extract_flexible(text: str) -> str | None:
    if not text:
        return None
    m = _HASH_RE.search(text)
    if m:
        return normalize_number(m.group(1))
    m = _ANS_RE.search(text)
    if m:
        return normalize_number(m.group(1))
    nums = _LAST_NUM_RE.findall(text.replace(",", ""))
    if nums:
        return normalize_number(nums[-1])
    return None


def score_gsm8k(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Each row: {id, gold, output} where gold is the numeric answer string."""
    correct = 0
    details = []
    for r in rows:
        gold = normalize_number(str(r["gold"]))
        pred = extract_flexible(str(r.get("output", "")))
        ok = pred is not None and pred == gold
        correct += int(ok)
        details.append({"id": r.get("id"), "gold": gold, "pred": pred, "pass": ok})
    n = len(rows)
    return {
        "metric": "accuracy",
        "accuracy": (correct / n) if n else 0.0,
        "n": n,
        "correct": correct,
        "protocol": "gsm8k_flexible_extract",
        "details": details,
    }
