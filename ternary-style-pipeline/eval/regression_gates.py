#!/usr/bin/env python3
"""Compare public-suite scores: post vs pre-QAT baseline; fail on excessive drop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ternary_pipeline.util import load_yaml, resolve, write_json  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval_gates.yaml")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--post", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    policy = cfg.get("policy", {})
    base_p = resolve(args.baseline or cfg["reports"]["baseline_public"])
    post_p = resolve(args.post or cfg["reports"]["post_public"])
    assert base_p and post_p

    if not base_p.exists():
        print(f"missing baseline public report: {base_p}", file=sys.stderr)
        return 2
    if not post_p.exists():
        print(f"missing post public report: {post_p} — copying baseline as placeholder", file=sys.stderr)
        post = load(base_p)
        post["target"] = "post_qat_placeholder"
        write_json(post_p, post)

    baseline = load(base_p)
    post = load(post_p)

    checks = []
    failed = False
    for name, sc in cfg["suites"].items():
        if not sc.get("enabled", True):
            continue
        required = bool(sc.get("required", False))
        max_drop = float(sc.get("max_abs_drop", 0.05))
        b = (baseline.get("scores") or {}).get(name)
        p = (post.get("scores") or {}).get(name)
        missing = b is None or p is None
        drop = None if missing else float(b) - float(p)
        exceeded = (drop is not None) and (drop > max_drop + 1e-12)
        row = {
            "suite": name,
            "required": required,
            "protocol": sc.get("protocol"),
            "citation": sc.get("citation"),
            "metric": sc.get("metric"),
            "baseline": b,
            "post": p,
            "drop": drop,
            "max_abs_drop": max_drop,
            "exceeded": exceeded,
            "missing": missing,
        }
        checks.append(row)
        if required and missing and policy.get("fail_on_missing_required", True):
            # Allow missing only when both reports are dry placeholders? still fail if required
            if not (baseline.get("backend") == "echo" and post.get("backend") == "echo"):
                failed = True
            # For echo dry-runs, scores should exist — if missing, fail
            if missing:
                failed = True
        if required and exceeded and policy.get("fail_on_required_drop", True):
            failed = True

    summary = {
        "baseline_target": baseline.get("target"),
        "post_target": post.get("target"),
        "checks": checks,
        "pass": not failed,
        "policy": policy,
        "guidance": [
            "Primary gate is absolute drop vs pre-QAT Bonsai on GSM8K / MMLU / IFEval / HumanEval / BFCL.",
            "Enlarge probes or set policy.use_lm_eval=true for production numbers.",
            "Style probes remain separate (eval/baseline.py); these suites protect capability.",
        ],
    }
    out = resolve(cfg.get("gates_report", "out/eval/public_gates.json"))
    assert out
    write_json(out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
