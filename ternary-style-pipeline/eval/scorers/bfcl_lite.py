"""BFCL-lite AST / exact tool-call match (Berkeley Function-Calling Leaderboard style)."""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_call(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    # Prefer fenced json
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    blob = fence.group(1) if fence else text
    m = _JSON_RE.search(blob)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None
    # Normalize OpenAI / generic shapes
    if "function" in obj and isinstance(obj["function"], dict):
        name = obj["function"].get("name")
        args = obj["function"].get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        return {"name": name, "arguments": args or {}}
    if "name" in obj:
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        return {"name": obj["name"], "arguments": args}
    return None


def _norm_args(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k in sorted(args.keys()):
        v = args[k]
        if isinstance(v, dict):
            out[k] = _norm_args(v)
        else:
            out[k] = v
    return out


def score_bfcl_lite(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Each row: {id, gold: {name, arguments}, output}."""
    exact = 0
    name_ok = 0
    details = []
    for r in rows:
        gold = r["gold"]
        pred = _parse_call(str(r.get("output", "")))
        g_name = gold.get("name")
        g_args = _norm_args(gold.get("arguments") or {})
        if pred is None:
            details.append({"id": r.get("id"), "pass": False, "pred": None})
            continue
        p_args = _norm_args(pred.get("arguments") or {})
        n_ok = pred.get("name") == g_name
        e_ok = n_ok and p_args == g_args
        name_ok += int(n_ok)
        exact += int(e_ok)
        details.append(
            {
                "id": r.get("id"),
                "pass": e_ok,
                "name_ok": n_ok,
                "pred": pred,
                "gold": {"name": g_name, "arguments": g_args},
            }
        )
    n = len(rows)
    return {
        "metric": "ast_exact_accuracy",
        "ast_exact_accuracy": (exact / n) if n else 0.0,
        "name_accuracy": (name_ok / n) if n else 0.0,
        "n": n,
        "exact": exact,
        "protocol": "bfcl_ast_exact",
        "details": details,
    }
