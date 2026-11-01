"""Track ternary code-flip rate vs a step-0 snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _ternary_codes(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    w = weight.detach().float()
    if w.shape[-1] % group_size:
        raise ValueError(f"last dim {tuple(w.shape)} not divisible by {group_size}")
    wg = w.reshape(*w.shape[:-1], w.shape[-1] // group_size, group_size)
    a = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.round(wg / a).clamp(-1, 1).to(torch.int8).reshape_as(w)


@torch.no_grad()
def snapshot_codes(model: torch.nn.Module, group_size: int = 128) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls not in ("TernaryLinear", "TernaryEmbedding"):
            continue
        w = getattr(mod, "weight", None)
        if w is None:
            continue
        gs = int(getattr(mod, "group_size", group_size))
        out[name] = _ternary_codes(w, gs).cpu()
    return out


@torch.no_grad()
def flip_rate(
    model: torch.nn.Module,
    baseline: dict[str, torch.Tensor],
    group_size: int = 128,
) -> dict[str, Any]:
    total = 0
    flipped = 0
    per_module: dict[str, float] = {}
    for name, mod in model.named_modules():
        if name not in baseline:
            continue
        w = getattr(mod, "weight", None)
        if w is None:
            continue
        gs = int(getattr(mod, "group_size", group_size))
        cur = _ternary_codes(w, gs).cpu()
        base = baseline[name]
        if cur.shape != base.shape:
            continue
        diff = (cur != base).sum().item()
        n = cur.numel()
        total += n
        flipped += int(diff)
        per_module[name] = diff / n if n else 0.0
    rate = (flipped / total) if total else 0.0
    return {
        "flip_rate": rate,
        "flipped": flipped,
        "total": total,
        "per_module_top": sorted(per_module.items(), key=lambda x: -x[1])[:8],
    }


def save_snapshot(path: str | Path, snap: dict[str, torch.Tensor]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snap, p)


def load_snapshot(path: str | Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def append_log(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def band_status(rate: float, cfg: dict[str, Any]) -> str:
    if rate < float(cfg.get("warn_if_near_zero", 0.0005)):
        return "too_low_scale_only"
    lo = float(cfg.get("target_min", 0.003))
    hi = float(cfg.get("target_max", 0.02))
    if rate < lo:
        return "below_band"
    if rate > hi:
        return "above_band_overfit_risk"
    return "in_band"
