#!/usr/bin/env python3
"""Phase 1 — generate style distillation JSONL from the teacher."""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ternary_pipeline.util import load_yaml, read_jsonl, resolve, write_jsonl  # noqa: E402


def openai_sample(messages: list[dict[str, Any]], cfg: dict[str, Any], temperature: float) -> str:
    payload = {
        "model": cfg["model_id"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": cfg.get("max_new_tokens", 1024),
    }
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.get('api_key', 'lm-studio')}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def echo_sample(messages: list[dict[str, Any]], temperature: float) -> str:
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    return f"[teacher-echo T={temperature}] {user}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/distill.yaml")
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--dry-run", action="store_true", help="write echo traces without calling teacher")
    args = ap.parse_args()

    dcfg = load_yaml(args.config)
    models = load_yaml(args.models)
    teacher = models["teacher"]

    prompts = read_jsonl(dcfg["prompts_file"])
    temps = list(dcfg.get("generation", {}).get("temperatures") or teacher.get("temperature_mix") or [0.7])
    max_traces = int(dcfg.get("generation", {}).get("max_traces", 200))
    spp = int(dcfg.get("generation", {}).get("samples_per_prompt", 1))
    sys_override = dcfg.get("generation", {}).get("system_prompt")

    rows: list[dict[str, Any]] = []
    rng = random.Random(0)
    for p in prompts:
        if len(rows) >= max_traces:
            break
        messages = list(p["messages"])
        if sys_override:
            if messages and messages[0].get("role") == "system":
                messages[0] = {"role": "system", "content": sys_override}
            else:
                messages = [{"role": "system", "content": sys_override}] + messages
        for _ in range(spp):
            if len(rows) >= max_traces:
                break
            t = rng.choice(temps)
            if args.dry_run or teacher.get("kind") == "echo":
                text = echo_sample(messages, t)
            else:
                text = openai_sample(messages, teacher, t)
            rows.append(
                {
                    "messages": messages + [{"role": "assistant", "content": text}],
                    "meta": {"temperature": t, "probe_id": p.get("id"), "source": "teacher_distill"},
                }
            )

    out = resolve(dcfg["output"]["teacher_traces"])
    assert out
    n = write_jsonl(out, rows)
    print(f"wrote {n} teacher traces → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
