"""HumanEval pass@1 (Chen et al.) — extract Python and optionally exec against tests."""

from __future__ import annotations

import re
import textwrap
import traceback
from typing import Any


_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_code(text: str, entry_point: str | None = None) -> str:
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        code = m.group(1)
    else:
        code = text
    code = textwrap.dedent(code).strip()
    # Drop leading prose lines before first def/class/import
    lines = code.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith(("def ", "class ", "import ", "from ")):
            start = i
            break
    code = "\n".join(lines[start:])
    if entry_point and f"def {entry_point}" not in code:
        # keep as-is; unit tests will fail clearly
        pass
    return code


def run_tests(code: str, test: str, entry_point: str, timeout_s: float = 2.0) -> tuple[bool, str]:
    """Execute candidate + canonical HumanEval-style assert tests in a restricted namespace."""
    # Minimal sandbox: no builtins beyond a safe subset. Not a security boundary —
    # disable with execute=false when untrusted.
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    ns: dict[str, Any] = {"__builtins__": safe_builtins}
    try:
        exec(code, ns, ns)  # noqa: S102 — intentional eval harness
        if entry_point not in ns:
            return False, f"missing entry_point {entry_point}"
        exec(test, ns, ns)  # noqa: S102
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"


def score_humaneval(rows: list[dict[str, Any]], execute: bool = True) -> dict[str, Any]:
    """Each row: {id, prompt, entry_point, test, output, canonical_solution?}."""
    passed = 0
    details = []
    for r in rows:
        entry = str(r.get("entry_point", "solution"))
        code = extract_code(str(r.get("output", "")), entry)
        if not execute:
            # Structural check only: entry point defined
            ok = f"def {entry}" in code
            msg = "syntax_only"
        else:
            ok, msg = run_tests(code, str(r.get("test", "")), entry)
        passed += int(ok)
        details.append(
            {
                "id": r.get("id"),
                "pass": ok,
                "message": msg,
                "code_preview": code[:200],
            }
        )
    n = len(rows)
    return {
        "metric": "pass_at_1",
        "pass_at_1": (passed / n) if n else 0.0,
        "n": n,
        "passed": passed,
        "execute": execute,
        "protocol": "humaneval_pass_at_1",
        "details": details,
    }
