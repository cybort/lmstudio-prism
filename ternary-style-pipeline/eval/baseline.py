#!/usr/bin/env python3
"""Phase 0 — run style/capability probes on teacher and untrained Bonsai.

Modes:
  --dry-run   validate probes + write empty baseline scaffolding (CI / no GPU)
  --target teacher|bonsai|both
  --backend hf|openai_compat|echo
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ternary_pipeline.util import (  # noqa: E402
    load_yaml,
    read_jsonl,
    resolve,
    write_json,
)


def _last_user(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def echo_complete(messages: list[dict[str, Any]]) -> str:
    """Deterministic stub for dry-run / wiring tests."""
    u = _last_user(messages)
    return f"[echo] {u[:240]}"


def openai_complete(messages: list[dict[str, Any]], base_url: str, api_key: str, model: str, **gen) -> str:
    import urllib.request
    import json

    payload = {
        "model": model,
        "messages": messages,
        "temperature": gen.get("temperature", 0.7),
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def hf_complete(model, tokenizer, messages: list[dict[str, Any]], **gen) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=gen.get("max_new_tokens", 512),
            do_sample=gen.get("temperature", 0.7) > 0,
            temperature=max(float(gen.get("temperature", 0.7)), 1e-5),
            top_p=gen.get("top_p", 0.9),
        )
    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def score_capability(text: str, expect: dict[str, Any] | None) -> bool:
    if not expect:
        return True
    low = text.lower()
    for needle in expect.get("contains_any", []):
        if needle.lower() in low:
            return True
    return False


def style_features(text: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z0-9_]+", text)
    chars = len(text)
    alnum = sum(ch.isalnum() or ch.isspace() for ch in text)
    return {
        "n_chars": float(chars),
        "n_words": float(len(words)),
        "alnum_ratio": (alnum / chars) if chars else 0.0,
        "mean_word_len": (sum(len(w) for w in words) / len(words)) if words else 0.0,
    }


def run_suite(
    name: str,
    probes: list[dict[str, Any]],
    completer,
    gen: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    cap_pass = 0
    cap_n = 0
    t0 = time.time()
    for p in probes:
        messages = p["messages"]
        text = completer(messages)
        feat = style_features(text)
        row = {
            "id": p.get("id"),
            "kind": p.get("kind", "style"),
            "output": text,
            "features": feat,
        }
        if "expect" in p:
            ok = score_capability(text, p.get("expect"))
            row["pass"] = ok
            cap_n += 1
            cap_pass += int(ok)
        rows.append(row)
    report = {
        "target": name,
        "n": len(rows),
        "capability_pass_rate": (cap_pass / cap_n) if cap_n else None,
        "capability_n": cap_n,
        "seconds": round(time.time() - t0, 3),
        "rows": rows,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--target", choices=["teacher", "bonsai", "both"], default="both")
    ap.add_argument("--backend", choices=["echo", "openai_compat", "hf"], default="echo")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    eval_cfg = load_yaml(args.config)
    models = load_yaml(args.models)
    out_dir = resolve(eval_cfg["output_dir"])
    assert out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    style = read_jsonl(eval_cfg["probes"]["style"])
    capability = read_jsonl(eval_cfg["probes"]["capability"])
    probes_style = style
    probes_all = style + capability
    gen = dict(eval_cfg.get("generation", {}))

    write_json(
        out_dir / "baseline_manifest.json",
        {
            "teacher": models.get("teacher"),
            "student": models.get("student"),
            "n_style": len(style),
            "n_capability": len(capability),
            "backend": "echo" if args.dry_run else args.backend,
            "pack_preference": models.get("pack_preference"),
        },
    )

    def make_completer(which: str):
        if args.dry_run or args.backend == "echo":
            return echo_complete
        if args.backend == "openai_compat":
            tcfg = models["teacher"] if which == "teacher" else models["student"]
            base = tcfg.get("base_url") or models["teacher"].get("base_url")
            key = tcfg.get("api_key") or "lm-studio"
            mid = tcfg.get("model_id")
            return lambda messages: openai_complete(messages, base, key, mid, **gen)
        # hf
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        mid = models["teacher"]["model_id"] if which == "teacher" else models["student"]["model_id"]
        local = models["student"].get("local_path") if which == "bonsai" else None
        src = local or mid
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        return lambda messages: hf_complete(model, tok, messages, **gen)

    targets = []
    if args.target in ("teacher", "both"):
        targets.append("teacher")
    if args.target in ("bonsai", "both"):
        targets.append("bonsai")

    for which in targets:
        completer = make_completer(which)
        style_report = run_suite(f"{which}_style", probes_style, completer, gen)
        cap_report = run_suite(f"{which}_capability", capability, completer, gen)
        report = {
            "target": which,
            "style": style_report,
            "capability": {
                "pass_rate": cap_report["capability_pass_rate"],
                "n": cap_report["capability_n"],
                "rows": cap_report["rows"],
            },
            "backend": "echo" if args.dry_run else args.backend,
        }
        path = out_dir / f"baseline_{which}.json"
        write_json(path, report)
        print(f"wrote {path} capability_pass_rate={report['capability']['pass_rate']}")

    print(f"baseline complete → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
