"""Minimal GGUF reader/writer that tolerates Prism types 142/143."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TensorInfo:
    name: str
    dims: list[int]  # as stored in GGUF (reversed vs numpy)
    dtype: int
    offset: int  # relative to tensor data section
    nbytes: int = 0


@dataclass
class GGUFFile:
    path: Path
    version: int
    kv: list[tuple[str, int, Any]] = field(default_factory=list)
    tensors: list[TensorInfo] = field(default_factory=list)
    data_offset: int = 0
    raw: bytes | None = None


class Rd:
    def __init__(self, b: bytes):
        self.b = b
        self.o = 0

    def raw(self, n: int) -> bytes:
        v = self.b[self.o : self.o + n]
        if len(v) != n:
            raise EOFError("truncated GGUF header")
        self.o += n
        return v

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.raw(4))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.raw(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.raw(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.raw(8))[0]

    def st(self) -> str:
        return self.raw(self.u64()).decode("utf-8")


def _rd_val(r: Rd, t: int):
    if t == 8:
        return r.st()
    if t in (0, 1, 7):
        return r.raw(1)[0]
    if t in (2, 3):
        return struct.unpack("<H", r.raw(2))[0]
    if t in (4, 5):
        return struct.unpack("<i", r.raw(4))[0]
    if t == 6:
        return struct.unpack("<f", r.raw(4))[0]
    if t in (10, 11):
        return struct.unpack("<q", r.raw(8))[0]
    if t == 12:
        return struct.unpack("<d", r.raw(8))[0]
    if t == 9:
        et = r.u32()
        n = r.u64()
        return (et, [_rd_val(r, et) for _ in range(n)])
    raise ValueError(f"unknown gguf value type {t}")


def _wr_str(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def _wr_raw(f, t: int, v):
    if t == 8:
        _wr_str(f, v)
    elif t in (0, 1, 7):
        f.write(struct.pack("<B", int(v)))
    elif t in (2, 3):
        f.write(struct.pack("<H", int(v)))
    elif t in (4, 5):
        f.write(struct.pack("<i", int(v)))
    elif t == 6:
        f.write(struct.pack("<f", float(v)))
    elif t in (10, 11):
        f.write(struct.pack("<q", int(v)))
    elif t == 12:
        f.write(struct.pack("<d", float(v)))
    elif t == 9:
        et, vals = v
        f.write(struct.pack("<I", et))
        f.write(struct.pack("<Q", len(vals)))
        for item in vals:
            _wr_raw(f, et, item)
    else:
        raise ValueError(f"unknown gguf value type {t}")


def tensor_nbytes(dtype: int, dims: list[int], layout_block: dict[int, tuple[int, int]] | None = None) -> int:
    """Return payload bytes for a tensor. layout_block maps dtype -> (group, block_bytes)."""
    n = 1
    for d in dims:
        n *= int(d)
    blocks = {
        0: (1, 4),    # F32
        1: (1, 2),    # F16
        24: (1, 2),   # BF16
        42: (64, 18), # Q2_0
        142: (128, 34),  # PQ2_0
        143: (128, 28),  # PTQ1_0
    }
    if layout_block:
        blocks.update(layout_block)
    if dtype not in blocks:
        # fall back: treat as opaque element size unknown — caller must set nbytes
        raise KeyError(f"unsupported dtype {dtype} for size calc")
    g, b = blocks[dtype]
    if n % g:
        raise ValueError(f"nelem {n} not divisible by group {g} for dtype {dtype}")
    return (n // g) * b


def read_gguf(path: str | Path, load_data: bool = True) -> GGUFFile:
    path = Path(path)
    data = path.read_bytes() if load_data else None
    # header only needs early bytes; for huge files mmap would be better — keep simple
    if data is None:
        with path.open("rb") as f:
            # read up to 256 MiB header+meta; tensors follow
            head = f.read(256 * 1024 * 1024)
        data = head
        full_load = False
    else:
        full_load = True

    r = Rd(data)
    magic = r.raw(4)
    if magic != b"GGUF":
        raise ValueError(f"not a GGUF file: {path}")
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    kv: list[tuple[str, int, Any]] = []
    for _ in range(n_kv):
        key = r.st()
        t = r.u32()
        val = _rd_val(r, t)
        kv.append((key, t, val))
    tensors: list[TensorInfo] = []
    for _ in range(n_tensors):
        name = r.st()
        n_dims = r.u32()
        dims = [r.u64() for _ in range(n_dims)]
        dtype = r.u32()
        offset = r.u64()
        try:
            nbytes = tensor_nbytes(dtype, dims)
        except KeyError:
            nbytes = 0
        tensors.append(TensorInfo(name=name, dims=dims, dtype=dtype, offset=offset, nbytes=nbytes))

    # align to 32
    while r.o % 32:
        r.o += 1
    data_offset = r.o

    # fill nbytes for unknown by next offsets if possible
    sorted_t = sorted(tensors, key=lambda t: t.offset)
    for i, t in enumerate(sorted_t):
        if t.nbytes:
            continue
        if i + 1 < len(sorted_t):
            t.nbytes = sorted_t[i + 1].offset - t.offset
        elif full_load:
            t.nbytes = len(data) - (data_offset + t.offset)

    g = GGUFFile(path=path, version=version, kv=kv, tensors=tensors, data_offset=data_offset)
    if full_load:
        g.raw = data
    return g


def get_tensor_bytes(g: GGUFFile, t: TensorInfo) -> bytes:
    if g.raw is None:
        with g.path.open("rb") as f:
            f.seek(g.data_offset + t.offset)
            return f.read(t.nbytes)
    return g.raw[g.data_offset + t.offset : g.data_offset + t.offset + t.nbytes]


def write_gguf(
    path: str | Path,
    version: int,
    kv: list[tuple[str, int, Any]],
    tensors: list[tuple[TensorInfo, bytes]],
) -> None:
    """Write a GGUF with provided tensor payloads (already packed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # compute offsets with 32-byte alignment between tensors? llama.cpp aligns
    # the data section start; individual tensor offsets are relative and typically
    # densely packed with padding to alignment.
    ALIGN = 32
    offsets: list[int] = []
    cur = 0
    payloads: list[bytes] = []
    infos: list[TensorInfo] = []
    for info, blob in tensors:
        pad = (ALIGN - (cur % ALIGN)) % ALIGN
        cur += pad
        offsets.append(cur)
        payloads.append(blob)
        infos.append(
            TensorInfo(
                name=info.name,
                dims=list(info.dims),
                dtype=info.dtype,
                offset=cur,
                nbytes=len(blob),
            )
        )
        cur += len(blob)

    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<Q", len(infos)))
        f.write(struct.pack("<Q", len(kv)))
        for key, t, val in kv:
            _wr_str(f, key)
            f.write(struct.pack("<I", t))
            _wr_raw(f, t, val)
        for info in infos:
            _wr_str(f, info.name)
            f.write(struct.pack("<I", len(info.dims)))
            for d in info.dims:
                f.write(struct.pack("<Q", int(d)))
            f.write(struct.pack("<I", int(info.dtype)))
            f.write(struct.pack("<Q", int(info.offset)))
        # align data section
        while f.tell() % ALIGN:
            f.write(b"\x00")
        data_start = f.tell()
        # write with padding holes
        cursor = 0
        for info, blob in zip(infos, payloads):
            while cursor < info.offset:
                f.write(b"\x00")
                cursor += 1
            f.write(blob)
            cursor += len(blob)
        _ = data_start  # retained for clarity / future mmap writers


def set_kv(kv: list[tuple[str, int, Any]], key: str, t: int, val: Any) -> list[tuple[str, int, Any]]:
    out: list[tuple[str, int, Any]] = []
    found = False
    for k, tt, vv in kv:
        if k == key:
            out.append((k, t, val))
            found = True
        else:
            out.append((k, tt, vv))
    if not found:
        out.append((key, t, val))
    return out
