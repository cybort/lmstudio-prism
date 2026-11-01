#!/usr/bin/env python3
"""Run publicly recognized capability suites (smoke subsets + optional lm-eval).

Protocols mirror GSM8K / MMLU / IFEval / HumanEval / BFCL / C-Eval / MT-Bench.
Use this before and after QAT to catch capability regressions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scorers import SCORERS  # noqa: E402
from ternary_pipeline.util import load_yaml, read_jsonl, resolve, write_json  # noqa: E402


def echo_complete(messages: list[dict[str, Any]], row: dict[str, Any], suite: str) -> str:
    """Deterministic answers so dry-run gates exercise scorers end-to-end."""
    if suite == "gsm8k":
        return f"Reasoning...\n#### {row['gold']}"
    if suite in ("mmlu", "ceval"):
        return f"Answer: {row['gold']}"
    if suite == "humaneval":
        return f"```python\n{row.get('gold_solution', '')}\n```"
    if suite == "bfcl_lite":
        return json.dumps(row["gold"], ensure_ascii=False)
    if suite == "ifeval":
        # Satisfy each instruction kind with a canned reply
        parts = []
        for inst in row.get("instructions") or []:
            k = inst["kind"]
            if k == "bullet_count":
                parts = [f"- item {i+1}" for i in range(int(inst["n"]))]
                return "\n".join(parts)
            if k == "json_object":
                return '{"city": "Shanghai", "country": "CN"}'
            if k == "number_sentences":
                return "Tea is popular worldwide. It comes in many varieties."
            if k == "all_caps":
                return "THE SKY IS BLUE"
            if k == "all_lowercase":
                return "apple"
            if k == "include_keyword":
                return f"hello from {inst['keyword']} lab"
            if k == "end_with":
                return f"Back up daily. {inst['suffix']}"
            if k == "number_words":
                # exactly n english-ish tokens
                n = int(inst["n"])
                words = ["cats", "are", "soft", "warm", "quiet", "pets", "that", "nap", "often", "indoors"]
                return " ".join(words[:n])
        return "ok"
    if suite == "mt_bench_lite":
        return (
            "A binary search tree stores keys so left < node < right.\n"
            "Example: insert 5, then 3, then 7 yields root 5 with children 3 and 7.\n"
            "This keeps ordered lookup efficient on balanced trees."
        )
    return "[echo]"


def openai_complete(messages: list[dict[str, Any]], base_url: str, api_key: str, model: str, **gen) -> str:
    import urllib.request

    payload = {
        "model": model,
        "messages": messages,
        "temperature": gen.get("temperature", 0.0),
        "max_tokens": gen.get("max_new_tokens", 512),
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def run_suite(
    name: str,
    suite_cfg: dict[str, Any],
    completer: Callable[..., str],
    gen: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    probes_path = resolve(suite_cfg["probes"])
    assert probes_path and probes_path.exists(), f"missing probes: {probes_path}"
    rows_in = read_jsonl(probes_path)
    scored_rows = []
    max_tok = gen.get("max_new_tokens", 512)
    if name == "humaneval":
        max_tok = gen.get("max_new_tokens_code", max_tok)
    local_gen = {**gen, "max_new_tokens": max_tok}

    for row in rows_in:
        messages = list(row["messages"])
        if dry_run:
            text = echo_complete(messages, row, name)
        else:
            text = completer(messages, **local_gen)
        item = dict(row)
        item["output"] = text
        scored_rows.append(item)

    scorer = SCORERS[name]
    if name == "humaneval":
        result = scorer(scored_rows, execute=bool(suite_cfg.get("execute", True)))
    elif name == "mt_bench_lite":
        result = scorer(scored_rows, judge=str(suite_cfg.get("judge", "heuristic")))
    else:
        result = scorer(scored_rows)

    metric_key = suite_cfg.get("metric", result.get("metric"))
    primary = result.get(metric_key, result.get("accuracy", result.get("pass_at_1")))
    return {
        "suite": name,
        "enabled": True,
        "required": bool(suite_cfg.get("required", False)),
        "protocol": suite_cfg.get("protocol"),
        "citation": suite_cfg.get("citation"),
        "metric": metric_key,
        "score": primary,
        "max_abs_drop": suite_cfg.get("max_abs_drop"),
        "result": result,
        "n": result.get("n"),
        "dry_run": dry_run,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval_gates.yaml")
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--target", default="bonsai", help="label stored in report (bonsai|teacher|post_qat|...)")
    ap.add_argument("--out", default=None, help="report path override")
    ap.add_argument("--backend", choices=["echo", "openai_compat"], default="echo")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--suites", nargs="*", default=None, help="subset of suite names")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    models = load_yaml(args.models)
    gen = dict(cfg.get("generation", {}))
    dry = args.dry_run or args.backend == "echo"

    if dry:
        completer: Callable[..., str] = lambda messages, **k: ""  # unused; echo path used
    else:
        tcfg = models.get("student") if args.target != "teacher" else models.get("teacher")
        # Prefer teacher endpoint fields for openai_compat
        ep = models.get("teacher", {})
        base = (tcfg or {}).get("base_url") or ep.get("base_url")
        key = (tcfg or {}).get("api_key") or ep.get("api_key") or "lm-studio"
        mid = (tcfg or {}).get("model_id") or ep.get("model_id")

        def completer(messages, **k):  # type: ignore[no-redef]
            return openai_complete(messages, base, key, mid, **k)

    suite_names = args.suites or [k for k, v in cfg["suites"].items() if v.get("enabled", True)]
    reports = []
    for name in suite_names:
        sc = cfg["suites"][name]
        if not sc.get("enabled", True):
            continue
        print(f"[public] running {name} ...", flush=True)
        reports.append(run_suite(name, sc, completer, gen, dry_run=dry))

    out_path = resolve(args.out) if args.out else resolve(cfg["output_dir"]) / f"public_{args.target}.json"
    assert out_path
    payload = {
        "target": args.target,
        "backend": "echo" if dry else args.backend,
        "suites": {r["suite"]: r for r in reports},
        "scores": {r["suite"]: r["score"] for r in reports},
    }
    write_json(out_path, payload)
    print(json.dumps({"out": str(out_path), "scores": payload["scores"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
