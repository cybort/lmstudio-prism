from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path | None, base: Path | None = None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or ROOT) / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = resolve(path)
    assert p is not None
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {p}")
    return data


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = resolve(path)
    assert p is not None
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = resolve(path)
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str | Path, obj: Any) -> None:
    p = resolve(path)
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_messages_text(messages: list[dict[str, Any]]) -> Iterator[str]:
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            yield c


def ensure_dir(path: str | Path) -> Path:
    p = resolve(path)
    assert p is not None
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}
