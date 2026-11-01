"""C-Eval letter accuracy (Huang et al.) — same MCQ letter protocol as MMLU."""

from __future__ import annotations

from typing import Any

from .mmlu import extract_letter


def score_ceval_letter(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "protocol": "ceval_letter_accuracy",
        "details": details,
    }
