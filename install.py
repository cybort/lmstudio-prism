#!/usr/bin/env python3
"""Cross-platform entry that shells to install.ps1 on Windows.

Used by the lmstudio-js CLI wrapper (register-custom).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Install Prism as an LM Studio backend")
    ap.add_argument("--config", type=Path, default=ROOT / "config.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--path", type=str, help="Override local_path (sets mode=local)")
    ap.add_argument("--url", type=str, help="Override url (sets mode=url)")
    args = ap.parse_args()

    example = ROOT / "config.example.json"
    cfg_path = args.config
    if not cfg_path.is_file():
        shutil.copy(example, cfg_path)
        print(f"[lmstudio-prism] created {cfg_path}", flush=True)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if args.path:
        cfg["mode"] = "local"
        cfg["local_path"] = args.path
    if args.url:
        cfg["mode"] = "url"
        cfg["url"] = args.url
    if args.path or args.url:
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    if os.name == "nt":
        ps1 = ROOT / "install.ps1"
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(ps1),
            "-ConfigPath", str(cfg_path),
        ]
        if args.force:
            cmd.append("-Force")
        return subprocess.call(cmd)

    print("install.py currently shells to install.ps1 (Windows). On Unix, install manually.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
