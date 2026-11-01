"""IFEval-style verifiable instruction checks (Zhou et al. 2023).

Smoke subset implements a subset of constraint types used in the paper:
  - number_words / number_sentences
  - all_caps / all_lowercase
  - include_keyword / end_with
  - bullet_count / json_object
Strict = all instructions for a prompt pass; loose = allow minor formatting slack.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p for p in parts if p.strip()]


def check_instruction(inst: dict[str, Any], text: str, loose: bool = False) -> bool:
    kind = inst["kind"]
    t = text or ""
    if kind == "number_words":
        n = len(_words(t))
        target = int(inst["n"])
        return abs(n - target) <= (2 if loose else 0)
    if kind == "number_sentences":
        n = len(_sentences(t))
        target = int(inst["n"])
        return abs(n - target) <= (1 if loose else 0)
    if kind == "all_caps":
        letters = [c for c in t if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)
    if kind == "all_lowercase":
        letters = [c for c in t if c.isalpha()]
        return bool(letters) and all(c.islower() for c in letters)
    if kind == "include_keyword":
        kw = str(inst["keyword"])
        return kw.lower() in t.lower() if loose else kw in t
    if kind == "end_with":
        suf = str(inst["suffix"])
        return t.rstrip().endswith(suf)
    if kind == "bullet_count":
        bullets = re.findall(r"(?m)^\s*[-*•]\s+\S+", t)
        target = int(inst["n"])
        return abs(len(bullets) - target) <= (1 if loose else 0)
    if kind == "json_object":
        try:
            start = t.find("{")
            end = t.rfind("}")
            if start < 0 or end < 0:
                return False
            obj = json.loads(t[start : end + 1])
            return isinstance(obj, dict)
        except Exception:
            return False
    raise ValueError(f"unknown IFEval instruction kind: {kind}")


def score_ifeval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Each row: {id, instructions: [...], output}."""
    prompt_strict = prompt_loose = 0
    inst_strict_ok = inst_strict_n = 0
    inst_loose_ok = inst_loose_n = 0
    details = []
    for r in rows:
        insts = list(r.get("instructions") or [])
        text = str(r.get("output", ""))
        s_flags = [check_instruction(i, text, loose=False) for i in insts]
        l_flags = [check_instruction(i, text, loose=True) for i in insts]
        ps = all(s_flags) if s_flags else False
        pl = all(l_flags) if l_flags else False
        prompt_strict += int(ps)
        prompt_loose += int(pl)
        inst_strict_ok += sum(s_flags)
        inst_strict_n += len(s_flags)
        inst_loose_ok += sum(l_flags)
        inst_loose_n += len(l_flags)
        details.append(
            {
                "id": r.get("id"),
                "prompt_strict": ps,
                "prompt_loose": pl,
                "strict_flags": s_flags,
                "loose_flags": l_flags,
            }
        )
    n = len(rows)
    return {
        "metric": "prompt_loose_accuracy",
        "prompt_strict_accuracy": (prompt_strict / n) if n else 0.0,
        "prompt_loose_accuracy": (prompt_loose / n) if n else 0.0,
        "inst_strict_accuracy": (inst_strict_ok / inst_strict_n) if inst_strict_n else 0.0,
        "inst_loose_accuracy": (inst_loose_ok / inst_loose_n) if inst_loose_n else 0.0,
        "n": n,
        "protocol": "ifeval_strict_loose",
        "details": details,
    }
