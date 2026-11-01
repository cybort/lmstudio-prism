#!/usr/bin/env python3
"""Mix teacher distill traces with capability anchors (default 80/20)."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ternary_pipeline.util import load_yaml, read_jsonl, resolve, write_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/distill.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    distill_path = resolve(cfg["output"]["teacher_traces"])
    cap_out = resolve(cfg["output"]["capability"])
    mixed_out = resolve(cfg["output"]["mixed"])
    assert distill_path and cap_out and mixed_out

    if not distill_path.exists():
        if args.dry_run:
            # seed from example
            example = resolve("data/samples/teacher_traces.example.jsonl")
            assert example
            shutil.copy(example, distill_path)
            print(f"dry-run: seeded {distill_path} from example")
        else:
            raise SystemExit(f"missing distill file: {distill_path}")

    # capability
    cap_src = resolve(cfg["capability"]["source"])
    assert cap_src and cap_src.exists()
    cap_rows = read_jsonl(cap_src)
    for r in cap_rows:
        r.setdefault("meta", {})["source"] = "capability_anchor"
    write_jsonl(cap_out, cap_rows)

    distill = read_jsonl(distill_path)
    ratio_d = float(cfg["mix"]["distill_ratio"])
    ratio_c = float(cfg["mix"]["capability_ratio"])
    if abs(ratio_d + ratio_c - 1.0) > 1e-6:
        print(f"warning: ratios sum to {ratio_d + ratio_c}, normalizing")
        s = ratio_d + ratio_c
        ratio_d, ratio_c = ratio_d / s, ratio_c / s

    # Build mixed set sized to distill length (cap resampled as needed)
    n = len(distill)
    n_cap = max(1, int(round(n * ratio_c / max(ratio_d, 1e-9)))) if distill else len(cap_rows)
    rng = random.Random(int(cfg["mix"].get("shuffle_seed", 42)))
    caps = [cap_rows[i % len(cap_rows)] for i in range(n_cap)] if cap_rows else []
    # take distill proportionally
    n_dist = max(1, int(round((len(caps) * ratio_d / max(ratio_c, 1e-9))))) if caps else n
    dist_sample = distill[:n] if n <= n_dist else rng.sample(distill, n_dist)
    # Prefer keeping all distill when small; then add caps to hit ratio
    mixed = list(dist_sample) + list(caps)
    rng.shuffle(mixed)
    write_jsonl(mixed_out, mixed)
    print(
        f"mixed {len(mixed)} rows "
        f"(distill≈{len(dist_sample)}, capability≈{len(caps)}) → {mixed_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
