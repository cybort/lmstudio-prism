#!/usr/bin/env python3
"""Compare post-QAT / shipped model reports against teacher + baseline Bonsai.

Also folds in public capability gates (GSM8K / MMLU / IFEval / HumanEval / BFCL / …)
from eval/regression_gates.py when those reports exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ternary_pipeline.util import load_yaml, resolve, write_json  # noqa: E402


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_alnum(report: dict) -> float:
    rows = report.get("style", {}).get("rows") or []
    if not rows:
        return 0.0
    return sum(r["features"]["alnum_ratio"] for r in rows) / len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--post", default="out/eval/post_qat.json")
    ap.add_argument("--skip-public-gates", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    teacher_p = resolve(cfg["targets"]["baseline_teacher_report"])
    bonsai_p = resolve(cfg["targets"]["baseline_bonsai_report"])
    post_p = resolve(args.post)
    assert teacher_p and bonsai_p and post_p

    if not teacher_p.exists() or not bonsai_p.exists():
        print("missing baseline reports; run eval/baseline.py first", file=sys.stderr)
        return 2
    if not post_p.exists():
        print(f"missing post report {post_p}; creating placeholder from bonsai", file=sys.stderr)
        post = load(bonsai_p)
        post["target"] = "post_qat_placeholder"
        write_json(post_p, post)

    teacher = load(teacher_p)
    bonsai = load(bonsai_p)
    post = load(post_p)

    thr = float(cfg.get("scoring", {}).get("capability_pass_threshold", 0.7))
    max_drop = float(cfg.get("scoring", {}).get("tool_rate_max_drop", 0.05))

    b_rate = bonsai.get("capability", {}).get("pass_rate") or 0.0
    p_rate = post.get("capability", {}).get("pass_rate") or 0.0
    style_shift = abs(mean_alnum(post) - mean_alnum(teacher))
    local_pass = (b_rate - p_rate) <= max_drop

    public_gates = None
    public_pass = True
    if not args.skip_public_gates:
        gates_cfg = cfg.get("public_gates_config", "configs/eval_gates.yaml")
        rc = subprocess.call([sys.executable, str(ROOT / "eval" / "regression_gates.py"), "--config", gates_cfg])
        gates_path = resolve(cfg["targets"].get("public_gates", "out/eval/public_gates.json"))
        if gates_path and gates_path.exists():
            public_gates = load(gates_path)
            public_pass = bool(public_gates.get("pass", False))
        elif rc != 0:
            public_pass = False

    summary = {
        "teacher_capability_pass_rate": teacher.get("capability", {}).get("pass_rate"),
        "bonsai_capability_pass_rate": b_rate,
        "post_capability_pass_rate": p_rate,
        "capability_drop_vs_bonsai": b_rate - p_rate,
        "style_alnum_shift_vs_teacher": style_shift,
        "pass_capability_floor": local_pass,
        "pass_public_gates": public_pass,
        "pass": local_pass and public_pass,
        "public_gates": public_gates,
        "notes": [
            "Replace echo/backends with real completions before trusting win-rates.",
            f"Configured local capability threshold={thr} (informational).",
            "Public gates: GSM8K / MMLU / IFEval / HumanEval / BFCL (+ optional C-Eval, MT-Bench).",
        ],
    }
    out = resolve(cfg["output_dir"]) / "regression_summary.json"
    assert out
    write_json(out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())