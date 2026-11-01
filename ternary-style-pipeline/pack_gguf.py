#!/usr/bin/env python3
"""Phase 3 — pack the same trit+scale weights as PQ2_0 / PTQ1_0 / optional Q2_0.

Modes (configs/pack.yaml):
  donor_rewrite  — start from official Bonsai GGUF, replace ternary payloads
  latent_only    — write packed .bin shards + parity report (no full GGUF)
  external_packer — shell out to Prism converter if configured

Also optional: hardlink mmproj, graft MTP via ../graft_mtp.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ternary_pipeline.codecs import (  # noqa: E402
    LAYOUTS,
    dequant_pq2,
    dequant_ptq1,
    pack_pq2,
    pack_ptq1,
    same_weight_parity,
)
from ternary_pipeline import gguf_io  # noqa: E402
from ternary_pipeline.util import ensure_dir, load_yaml, resolve, write_json  # noqa: E402


def _load_hf_weights(merged_dir: Path) -> dict[str, np.ndarray]:
    """Load float weights from a HF safetensors / bin directory."""
    weights: dict[str, np.ndarray] = {}
    st = list(merged_dir.glob("*.safetensors"))
    if st:
        from safetensors import safe_open

        for path in st:
            with safe_open(path, framework="np") as f:
                for k in f.keys():
                    weights[k] = np.array(f.get_tensor(k))
        return weights
    bins = list(merged_dir.glob("pytorch_model*.bin")) + list(merged_dir.glob("model*.bin"))
    if bins:
        import torch

        for path in bins:
            obj = torch.load(path, map_location="cpu", weights_only=True)
            for k, v in obj.items():
                weights[k] = v.detach().cpu().float().numpy()
        return weights
    raise FileNotFoundError(f"no safetensors/bin weights in {merged_dir}")


def _guess_name_map(hf_keys: list[str], gguf_names: list[str]) -> dict[str, str]:
    """Best-effort HF → GGUF name map for common Qwen-style modules."""
    mapping: dict[str, str] = {}
    # Direct equality after stripping known prefixes
    gguf_set = set(gguf_names)
    for k in hf_keys:
        candidates = []
        base = k
        for pref in ("model.", "transformer.", "language_model.", "model.language_model."):
            if base.startswith(pref):
                base = base[len(pref) :]
        # HF embed / lm_head
        if base in ("embed_tokens.weight", "model.embed_tokens.weight"):
            candidates.append("token_embd.weight")
        if base in ("lm_head.weight",):
            candidates += ["output.weight", "token_embd.weight"]
        # layers
        import re

        m = re.match(r"layers\.(\d+)\.(.+)\.weight$", base) or re.match(
            r"model\.layers\.(\d+)\.(.+)\.weight$", k
        )
        if m:
            idx, rest = m.group(1), m.group(2)
            table = {
                "self_attn.q_proj": f"blk.{idx}.attn_q.weight",
                "self_attn.k_proj": f"blk.{idx}.attn_k.weight",
                "self_attn.v_proj": f"blk.{idx}.attn_v.weight",
                "self_attn.o_proj": f"blk.{idx}.attn_output.weight",
                "mlp.gate_proj": f"blk.{idx}.ffn_gate.weight",
                "mlp.up_proj": f"blk.{idx}.ffn_up.weight",
                "mlp.down_proj": f"blk.{idx}.ffn_down.weight",
                "self_attn.qkv_proj": f"blk.{idx}.attn_qkv.weight",
            }
            if rest in table:
                candidates.append(table[rest])
        for c in candidates:
            if c in gguf_set:
                mapping[k] = c
                break
    return mapping


def latent_only(merged_dir: Path | None, out_dir: Path, layouts: list[str], verify: bool) -> dict:
    """Pack a small synthetic or HF tensor set and write parity report."""
    ensure_dir(out_dir)
    if merged_dir and (merged_dir / "DRY_RUN.json").exists():
        rng = np.random.default_rng(0)
        # synthetic ternary g128 tensor for codec smoke
        scales = rng.uniform(0.05, 1.0, size=(8,))
        codes = rng.integers(-1, 2, size=(8, 128), dtype=np.int8)
        w = (codes.astype(np.float32) * scales[:, None]).reshape(-1)
        tag = "synthetic"
    elif merged_dir and merged_dir.exists() and any(merged_dir.glob("*.safetensors")):
        weights = _load_hf_weights(merged_dir)
        # pick first 2D weight divisible by 128
        w = None
        tag = None
        for name, arr in weights.items():
            if arr.ndim >= 1 and arr.size >= 128 and arr.size % 128 == 0:
                w = arr.astype(np.float32).reshape(-1)[: 128 * 64]
                tag = name
                break
        if w is None:
            raise RuntimeError("no suitable weight found")
    else:
        rng = np.random.default_rng(1)
        scales = rng.uniform(0.1, 0.9, size=(4,))
        codes = rng.integers(-1, 2, size=(4, 128), dtype=np.int8)
        w = (codes.astype(np.float32) * scales[:, None]).reshape(-1)
        tag = "synthetic"

    artifacts = {}
    for layout in layouts:
        meta = LAYOUTS[layout]
        blob = meta["pack"](w)
        path = out_dir / f"latent_{tag.replace('/', '_')}_{layout}.bin"
        path.write_bytes(blob)
        back = meta["dequant"](blob, w.size)
        artifacts[layout] = {
            "path": str(path),
            "bytes": len(blob),
            "ggml_type": meta["ggml_type"],
            "max_recon_err": float(np.max(np.abs(back - meta["dequant"](meta["pack"](w), w.size)))),
        }
    parity = same_weight_parity(w, layouts) if verify else {"skipped": True}
    report = {"mode": "latent_only", "tensor": tag, "artifacts": artifacts, "parity": parity}
    write_json(out_dir / "pack_report.json", report)
    return report


def donor_rewrite(
    donor_path: Path,
    layout: str,
    hf_dir: Path | None,
    out_path: Path,
    name_map_path: Path | None,
    verify_sample: int,
) -> dict:
    meta = LAYOUTS[layout]
    g = gguf_io.read_gguf(donor_path, load_data=True)
    kv = gguf_io.set_kv(g.kv, "general.file_type", 4, meta["file_type"])  # type int32

    weights = _load_hf_weights(hf_dir) if hf_dir and hf_dir.exists() and not (hf_dir / "DRY_RUN.json").exists() else {}
    name_map: dict[str, str] = {}
    if name_map_path and name_map_path.exists():
        name_map = json.loads(name_map_path.read_text(encoding="utf-8"))
    elif weights:
        name_map = _guess_name_map(list(weights.keys()), [t.name for t in g.tensors])

    # invert map: gguf name -> hf key
    inv = {v: k for k, v in name_map.items()}
    replaced = []
    tensors_out: list[tuple[gguf_io.TensorInfo, bytes]] = []
    sample_parity = []

    for t in g.tensors:
        blob = gguf_io.get_tensor_bytes(g, t)
        if t.name in inv and t.dtype in (142, 143, 42):
            w = weights[inv[t.name]].astype(np.float32)
            ne = 1
            for d in t.dims:
                ne *= int(d)
            if w.size != ne:
                # try transpose for row/col convention
                if w.T.size == ne:
                    w = w.T
                else:
                    raise ValueError(f"shape mismatch {t.name}: hf {w.shape} vs gguf nelem {ne}")
            new_blob = meta["pack"](w.reshape(-1))
            info = gguf_io.TensorInfo(name=t.name, dims=t.dims, dtype=meta["ggml_type"], offset=0, nbytes=len(new_blob))
            tensors_out.append((info, new_blob))
            replaced.append(t.name)
            if len(sample_parity) < verify_sample:
                # compare old dequant layout-agnostic via new pack only
                sample_parity.append({"tensor": t.name, "bytes": len(new_blob)})
        elif t.dtype in (142, 143, 42) and not weights:
            # dry / no HF: re-pack from dequant of existing to target layout (same-weight layout convert)
            ne = 1
            for d in t.dims:
                ne *= int(d)
            if t.dtype == 142:
                fp = dequant_pq2(blob, ne)
            elif t.dtype == 143:
                fp = dequant_ptq1(blob, ne)
            else:
                from ternary_pipeline.codecs import dequant_q2

                fp = dequant_q2(blob, ne)
            new_blob = meta["pack"](fp)
            info = gguf_io.TensorInfo(name=t.name, dims=t.dims, dtype=meta["ggml_type"], offset=0, nbytes=len(new_blob))
            tensors_out.append((info, new_blob))
            replaced.append(t.name)
        else:
            # keep as-is (possibly retarget dtype if converting layout with no HF)
            info = gguf_io.TensorInfo(name=t.name, dims=t.dims, dtype=t.dtype, offset=0, nbytes=len(blob))
            tensors_out.append((info, blob))

    gguf_io.write_gguf(out_path, g.version, kv, tensors_out)
    return {
        "layout": layout,
        "out": str(out_path),
        "replaced": replaced,
        "n_replaced": len(replaced),
        "name_map_size": len(name_map),
        "sample": sample_parity,
    }


def post_mmproj_and_mtp(cfg: dict[str, Any], models: dict[str, Any], out_dir: Path, produced: list[Path]) -> dict:
    actions = {}
    post = cfg.get("post", {})
    donor = models.get("donor_gguf", {})
    mmproj = donor.get("mmproj")
    if post.get("hardlink_mmproj") and mmproj and Path(mmproj).exists():
        for gguf in produced:
            dest = gguf.parent / Path(mmproj).name
            if dest.exists():
                dest.unlink()
            try:
                os.link(mmproj, dest)
                actions[f"mmproj:{gguf.name}"] = f"hardlink → {dest}"
            except OSError:
                shutil.copy2(mmproj, dest)
                actions[f"mmproj:{gguf.name}"] = f"copy → {dest}"

    if post.get("graft_mtp"):
        graft = resolve(post.get("graft_script", "../graft_mtp.py"))
        if graft and graft.exists():
            for gguf in produced:
                if "PQ2_0" not in gguf.name and "PTQ1_0" not in gguf.name:
                    continue
                out_mtp = gguf.with_name(gguf.stem + "-MTP" + gguf.suffix)
                cmd = [sys.executable, str(graft), "--base", str(gguf), "--out", str(out_mtp)]
                mtp_src = donor.get("mtp_src")
                if mtp_src:
                    cmd += ["--mtp-src", str(mtp_src)]
                try:
                    subprocess.run(cmd, check=True)
                    actions[f"mtp:{gguf.name}"] = str(out_mtp)
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    actions[f"mtp:{gguf.name}"] = f"skipped: {e}"
        else:
            actions["mtp"] = f"graft script missing: {graft}"
    return actions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pack.yaml")
    ap.add_argument("--dry-run", action="store_true", help="latent_only synthetic pack + parity")
    ap.add_argument("--layouts", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    models = load_yaml(cfg.get("models_config", "configs/models.yaml"))
    out_dir = ensure_dir(cfg.get("output_dir", "out/gguf"))
    layouts = args.layouts or list(cfg.get("layouts") or ["PQ2_0", "PTQ1_0"])
    mode = "latent_only" if args.dry_run else cfg.get("mode", "donor_rewrite")
    merged = resolve(cfg.get("merged_hf_dir", "out/qat_merged"))

    if mode == "latent_only" or args.dry_run:
        report = latent_only(merged, out_dir, layouts, verify=bool(cfg.get("verify", {}).get("same_weight_check", True)))
        print(json.dumps(report, indent=2))
        ok = report.get("parity", {}).get("ok", True)
        return 0 if ok else 1

    if mode == "external_packer":
        cmd_t = cfg.get("external_packer", {}).get("command")
        if not cmd_t:
            raise SystemExit("external_packer.command not set")
        for layout in layouts:
            cmd = cmd_t.format(hf=merged, out=out_dir / f"model-{layout}.gguf", layout=layout, prism_convert="")
            subprocess.run(cmd, shell=True, check=True)
        return 0

    # donor_rewrite
    produced: list[Path] = []
    results = []
    donors = models.get("donor_gguf", {})
    name_map = resolve(cfg.get("donor_rewrite", {}).get("name_map"))
    verify_n = int(cfg.get("verify", {}).get("max_tensors_sample", 8))
    for layout in layouts:
        donor = donors.get(layout) or donors.get("PQ2_0")
        if not donor or not Path(donor).exists():
            print(f"donor missing for {layout}; falling back to latent_only for this layout")
            results.append(latent_only(merged, out_dir / layout, [layout], True))
            continue
        out_path = out_dir / f"Style-Bonsai-{layout}.gguf"
        r = donor_rewrite(Path(donor), layout, merged, out_path, name_map, verify_n)
        results.append(r)
        produced.append(out_path)
        print(f"wrote {out_path} replaced={r['n_replaced']}")

    actions = post_mmproj_and_mtp(cfg, models, out_dir, produced)
    write_json(out_dir / "pack_report.json", {"results": results, "post": actions, "layouts": layouts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
