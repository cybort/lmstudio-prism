#!/usr/bin/env python3
"""Phase 2 — ternary_QAT + LoRA on unpacked Bonsai-2 (group 128).

Requires: torch, transformers, peft, ternary_QAT
Use --dry-run to validate config + data wiring without loading weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ternary_pipeline.util import ensure_dir, load_yaml, read_jsonl, resolve, write_json  # noqa: E402


def _format_example(messages: list[dict[str, Any]], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    parts = []
    for m in messages:
        parts.append(f"{m.get('role', 'user').upper()}: {m.get('content', '')}")
    return "\n".join(parts)


def dry_run(cfg: dict[str, Any], models: dict[str, Any]) -> int:
    train_file = resolve(cfg["train_file"])
    assert train_file and train_file.exists(), f"missing {train_file}"
    rows = read_jsonl(train_file)
    out = ensure_dir(cfg["output_dir"])
    report = {
        "dry_run": True,
        "n_train_rows": len(rows),
        "student": models.get("student"),
        "lora": cfg.get("lora"),
        "optim": {
            "learning_rate": cfg.get("optim", {}).get("learning_rate"),
            "num_train_epochs": cfg.get("optim", {}).get("num_train_epochs"),
        },
        "code_flip_targets": cfg.get("code_flip"),
        "notes": [
            "swap_linear(..., TernaryConfig(group_size=128)) before PEFT",
            "LR band 5e-4..1e-3; monitor code-flip 0.3%–2% (~0.7% sweet spot)",
            "1–2 epoch early stop on style/tool metrics",
        ],
    }
    write_json(out / "dry_run_plan.json", report)
    # stub merged dir marker
    merged = ensure_dir(cfg.get("merge", {}).get("final_dir", "out/qat_merged"))
    write_json(merged / "DRY_RUN.json", {"ok": True, "message": "replace by real merge output"})
    print(json.dumps(report, indent=2))
    return 0


def train(cfg: dict[str, Any], models: dict[str, Any]) -> int:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model
    from ternary import TernaryConfig, swap_linear, ternarize_lora_params, reternarize_merged_linears

    from ternary_pipeline import code_flip

    student = models["student"]
    src = student.get("local_path") or student["model_id"]
    group_size = int(cfg.get("ternary", {}).get("group_size") or student.get("group_size") or 128)

    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        src,
        torch_dtype=torch.bfloat16 if cfg.get("optim", {}).get("bf16", True) else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    swap_linear(model, TernaryConfig(group_size=group_size))

    lcfg = cfg["lora"]
    peft_cfg = LoraConfig(
        r=int(lcfg["r"]),
        lora_alpha=int(lcfg["lora_alpha"]),
        lora_dropout=float(lcfg.get("lora_dropout", 0.0)),
        bias=lcfg.get("bias", "none"),
        target_modules=list(lcfg["target_modules"]),
        modules_to_save=list(lcfg.get("modules_to_save") or []),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    out_dir = ensure_dir(cfg["output_dir"])
    flip_cfg = cfg.get("code_flip", {})
    snap = code_flip.snapshot_codes(model, group_size)
    snap_path = resolve(flip_cfg.get("snapshot_path", out_dir / "code_flip_step0.pt"))
    assert snap_path
    code_flip.save_snapshot(snap_path, snap)
    flip_log = resolve(flip_cfg.get("log_path", out_dir / "code_flip.jsonl"))
    assert flip_log
    if flip_log.exists():
        flip_log.unlink()

    train_rows = read_jsonl(cfg["train_file"])
    max_len = int(cfg.get("optim", {}).get("max_seq_length", 4096))

    class JsonlDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            text = _format_example(self.rows[idx]["messages"], tokenizer)
            toks = tokenizer(text, truncation=True, max_length=max_len, padding=False)
            toks["labels"] = list(toks["input_ids"])
            return toks

    dataset = JsonlDataset(train_rows)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    optim = cfg["optim"]

    class FlipCallback:
        """Lightweight callback duck-typed for Trainer."""

        def __init__(self):
            self.trainer = None

        def on_train_begin(self, args, state, control, **kwargs):
            return control

        def on_step_end(self, args, state, control, **kwargs):
            every = int(flip_cfg.get("every_n_steps", 50))
            if state.global_step == 0 or state.global_step % every:
                return control
            stats = code_flip.flip_rate(model, snap, group_size)
            status = code_flip.band_status(stats["flip_rate"], flip_cfg)
            row = {
                "step": state.global_step,
                "flip_rate": stats["flip_rate"],
                "status": status,
                "flipped": stats["flipped"],
                "total": stats["total"],
            }
            code_flip.append_log(flip_log, row)
            print(f"[code-flip] step={state.global_step} rate={stats['flip_rate']:.4%} status={status}")
            return control

        # Trainer looks for Event methods via CallbackHandler — wrap as TrainerCallback if available
        def __getattr__(self, name):
            if name.startswith("on_"):
                return lambda *a, **k: None
            raise AttributeError(name)

    try:
        from transformers import TrainerCallback

        class _CB(TrainerCallback):
            def __init__(self, inner):
                self.inner = inner

            def on_step_end(self, args, state, control, **kwargs):
                return self.inner.on_step_end(args, state, control, **kwargs)

        callbacks = [_CB(FlipCallback())]
    except Exception:
        callbacks = []

    args_tr = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=float(optim["learning_rate"]),
        weight_decay=float(optim.get("weight_decay", 0.0)),
        warmup_ratio=float(optim.get("warmup_ratio", 0.03)),
        lr_scheduler_type=optim.get("lr_scheduler_type", "cosine"),
        max_grad_norm=float(optim.get("max_grad_norm", 1.0)),
        num_train_epochs=float(optim.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(optim.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(optim.get("gradient_accumulation_steps", 8)),
        bf16=bool(optim.get("bf16", True)),
        logging_steps=int(optim.get("logging_steps", 10)),
        save_steps=int(optim.get("save_steps", 200)),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()

    # merge path
    if cfg.get("merge", {}).get("merge_on_save", True):
        n_lora = ternarize_lora_params(model, group_size=group_size)
        print(f"ternarized {n_lora} LoRA param tensors")
        model = model.merge_and_unload()
        n_lin = reternarize_merged_linears(model)
        print(f"reternarized {n_lin} merged linears/embeddings")
        final_dir = ensure_dir(cfg["merge"]["final_dir"])
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        final_stats = code_flip.flip_rate(model, snap, group_size)
        write_json(final_dir / "code_flip_final.json", final_stats)
        print(f"merged model → {final_dir} final_flip={final_stats['flip_rate']:.4%}")
    else:
        model.save_pretrained(out_dir / "adapter")
        tokenizer.save_pretrained(out_dir / "adapter")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/lora_qat.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    models = load_yaml(cfg.get("student_config", "configs/models.yaml"))
    if args.dry_run:
        return dry_run(cfg, models)
    return train(cfg, models)


if __name__ == "__main__":
    raise SystemExit(main())
