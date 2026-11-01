#!/usr/bin/env python3
"""Prism llama-server smoke test for packed PQ2_0 / PTQ1_0 GGUFs."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ternary_pipeline.util import load_yaml, resolve  # noqa: E402


def find_llama_server(explicit: str | None) -> Path | None:
    if explicit and Path(explicit).exists():
        return Path(explicit)
    home = Path.home() / ".lmstudio" / "extensions" / "backends"
    if home.exists():
        for p in home.rglob("llama-server*"):
            if p.is_file():
                return p
    for name in ("llama-server", "llama-server.exe"):
        found = shutil_which(name)
        if found:
            return Path(found)
    return None


def shutil_which(cmd: str) -> str | None:
    from shutil import which

    return which(cmd)


def port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def gibberish_ratio(text: str) -> float:
    if not text:
        return 1.0
    good = sum(ch.isalnum() or ch.isspace() or ch in ".,!?;:'\"-()[]{}" for ch in text)
    return 1.0 - good / len(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/lms.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    gguf_dir = resolve(cfg.get("gguf_dir", "out/gguf"))
    assert gguf_dir
    models = sorted(gguf_dir.glob("*.gguf"))
    # Prefer language models over mmproj
    models = [m for m in models if "mmproj" not in m.name.lower()]

    report = {
        "dry_run": args.dry_run,
        "gguf_dir": str(gguf_dir),
        "models_found": [m.name for m in models],
        "runtime": cfg.get("runtime"),
        "warnings": [
            "Never combine --kv-mean-center with kCache=q8_0",
        ],
    }

    if args.dry_run or not models:
        report["status"] = "skipped" if not models else "dry_run_ok"
        print(json.dumps(report, indent=2))
        return 0

    server = find_llama_server(cfg.get("runtime", {}).get("llama_server"))
    if not server:
        report["status"] = "llama_server_not_found"
        print(json.dumps(report, indent=2))
        return 2

    host = cfg["runtime"].get("host", "127.0.0.1")
    port = int(cfg["runtime"].get("port", 8765))
    results = []
    for model in models[:2]:
        if port_open(host, port):
            report.setdefault("notes", []).append(f"port {port} busy; skip kill — user must free it")
            break
        cmd = [
            str(server),
            "-m",
            str(model),
            "--host",
            host,
            "--port",
            str(port),
            "-c",
            str(cfg["runtime"].get("ctx", 8192)),
            "-ngl",
            str(cfg["runtime"].get("gpu_layers", 999)),
            "--parallel",
            str(cfg["runtime"].get("parallel", 1)),
            "-ctk",
            str(cfg["runtime"].get("k_cache", "q8_0")),
            "-ctv",
            str(cfg["runtime"].get("v_cache", "q4_0")),
        ]
        # intentionally omit --kv-mean-center
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ready = False
        t0 = time.time()
        log = []
        try:
            while time.time() - t0 < 120:
                if proc.poll() is not None:
                    break
                if port_open(host, port):
                    ready = True
                    break
                time.sleep(1)
            text = ""
            err = None
            if ready:
                payload = {
                    "prompt": cfg.get("smoke", {}).get("prompt", "Say hello."),
                    "n_predict": 32,
                    "temperature": 0.0,
                }
                try:
                    req = urllib.request.Request(
                        f"http://{host}:{port}/completion",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                    text = body.get("content") or body.get("choices", [{}])[0].get("text", "")
                except Exception as e:  # noqa: BLE001
                    err = str(e)
            results.append(
                {
                    "model": model.name,
                    "ready": ready,
                    "exit": proc.poll(),
                    "text": text,
                    "gibberish_ratio": gibberish_ratio(text),
                    "error": err,
                    "ok": bool(
                        ready
                        and text
                        and len(text) >= int(cfg.get("smoke", {}).get("min_chars", 8))
                        and gibberish_ratio(text)
                        <= float(cfg.get("smoke", {}).get("reject_gibberish_ratio", 0.35))
                    ),
                }
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    report["results"] = results
    report["status"] = "ok" if results and all(r.get("ok") for r in results) else "fail"
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
