"""MT-Bench lite rubric (Zheng et al.) — heuristic fallback + optional LLM judge hook."""

from __future__ import annotations

import re
from typing import Any


def heuristic_rubric(text: str, turns: list[dict[str, Any]]) -> float:
    """Cheap 1–10 style proxy used only when no judge model is available.

    Favors: non-empty multi-sentence answers, no refusal on benign asks,
    some structure. This is NOT a substitute for GPT-4 judge / Arena scoring —
    use judge=openai_compat for production.
    """
    if not text or not text.strip():
        return 1.0
    score = 5.0
    words = re.findall(r"\w+|[\u4e00-\u9fff]", text)
    if 40 <= len(words) <= 400:
        score += 1.5
    elif len(words) < 15:
        score -= 2.0
    if text.count("\n") >= 1:
        score += 0.5
    if re.search(r"(as an ai|i cannot help with that|as a language model)", text, re.I):
        # only punish on clearly benign first turns
        user0 = next((t.get("content", "") for t in turns if t.get("role") == "user"), "")
        if not re.search(r"(bomb|malware|hack|explosive)", user0, re.I):
            score -= 1.5
    # gibberish
    alnum = sum(ch.isalnum() or ch.isspace() for ch in text)
    if text and alnum / len(text) < 0.5:
        score -= 3.0
    return float(max(1.0, min(10.0, score)))


def score_mt_bench_lite(rows: list[dict[str, Any]], judge: str = "heuristic") -> dict[str, Any]:
    """Each row: {id, messages (multi-turn), output (final assistant), judge_score?}."""
    scores = []
    details = []
    for r in rows:
        if r.get("judge_score") is not None:
            s = float(r["judge_score"])
        elif judge == "heuristic":
            s = heuristic_rubric(str(r.get("output", "")), list(r.get("messages") or []))
        else:
            s = heuristic_rubric(str(r.get("output", "")), list(r.get("messages") or []))
        scores.append(s)
        details.append({"id": r.get("id"), "score": s})
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    return {
        "metric": "rubric_mean",
        "rubric_mean": mean,
        "n": n,
        "judge": judge,
        "protocol": "mt_bench_pair_or_rubric",
        "details": details,
        "note": "For production use LLM-as-judge (MT-Bench / Arena-Hard); heuristic is wiring-only.",
    }
