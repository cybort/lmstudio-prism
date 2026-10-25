#!/usr/bin/env python3
"""Register an installed backend folder in LM Studio's internal-engine-index.json."""
from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-dir", required=True, type=Path)
    ap.add_argument("--donor-name", required=True)
    ap.add_argument("--donor-version", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--version", required=True)
    args = ap.parse_args()

    backend_dir = args.backend_dir.resolve()
    manifest_path = backend_dir / "backend-manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest["name"] = args.name
    manifest["version"] = args.version

    home = Path.home() / ".lmstudio"
    idx_path = home / ".internal" / "internal-engine-index.json"
    if not idx_path.is_file():
        raise SystemExit(f"missing engine index: {idx_path}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = idx_path.with_suffix(f".json.bak-{stamp}")
    shutil.copy2(idx_path, backup)

    data = json.loads(idx_path.read_text(encoding="utf-8"))
    entries = data.get("json") or []
    donor = next(
        (
            e
            for e in entries
            if (e.get("manifest") or {}).get("name") == args.donor_name
            and (e.get("manifest") or {}).get("version") == args.donor_version
        ),
        None,
    )
    if donor is None:
        raise SystemExit(f"donor {args.donor_name}@{args.donor_version} not in engine index")

    # Drop any previous registration of this name@version
    entries = [
        e
        for e in entries
        if not (
            (e.get("manifest") or {}).get("name") == args.name
            and (e.get("manifest") or {}).get("version") == args.version
        )
    ]

    neo = deepcopy(donor)
    neo["manifest"] = manifest
    neo["dirPath"] = str(backend_dir)
    neo["libLmStudioPath"] = str(backend_dir / Path(donor["libLmStudioPath"]).name)
    if neo.get("launchInfo") and neo["launchInfo"].get("engineLibPath"):
        neo["launchInfo"] = {
            **neo["launchInfo"],
            "engineLibPath": str(backend_dir / Path(donor["launchInfo"]["engineLibPath"]).name),
        }
    # Keep donor displayData / hardwareSurvey / envVars (vendor PATH still valid)

    entries.append(neo)
    data["json"] = entries
    idx_path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[lmstudio-prism] registered {args.name}@{args.version} in {idx_path}")
    print(f"[lmstudio-prism] index backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
