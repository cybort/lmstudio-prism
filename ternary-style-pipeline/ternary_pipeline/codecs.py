"""PQ2_0 / PTQ1_0 / Q2_0 block codecs for same-weight ternary packing.

Layouts match Prism / llama.cpp references:
  PQ2_0  (type 142): 34 B / 128 w  — fp16 d + 32 B of 2-bit codes (code = trit+1)
  PTQ1_0 (type 143): 28 B / 128 w  — 24 B qs + 2 B qh (base-3) + fp16 d
  Q2_0   (type  42): 18 B /  64 w  — fp16 d + 16 B of 2-bit codes (group-64)

Encoding for 2-bit: stored digit = trit + 1  →  {0,1,2} for {-1,0,+1}
(Q2_0 quant path historically clips to [-1, 2]; for already-ternary inputs we
only emit {0,1,2}.)
"""

from __future__ import annotations

import struct
from typing import Iterable

import numpy as np

GGML_TYPE_Q2_0 = 42
GGML_TYPE_PQ2_0 = 142
GGML_TYPE_PTQ1_0 = 143

LLAMA_FTYPE_MOSTLY_Q2_0 = 41
LLAMA_FTYPE_MOSTLY_PQ2_0 = 141
LLAMA_FTYPE_MOSTLY_PTQ1_0 = 143

QK_PQ2 = 128
PQ2_BYTES = 34

QK_PTQ1 = 128
PTQ1_QS = 24
PTQ1_QH = 2
PTQ1_BYTES = 28

QK_Q2 = 64
Q2_BYTES = 18

POW3 = (1, 3, 9, 27, 81)
PTQ1_STAGES = (32, 16, 8)


def _f16_bytes(x: float) -> bytes:
    return np.float16(x).tobytes()


def _read_f16(b: bytes | memoryview) -> float:
    return float(np.frombuffer(bytes(b), dtype=np.float16)[0])


def ternarize_group(vals: np.ndarray) -> tuple[np.ndarray, float]:
    """vals (K,) float → codes in {-1,0,+1}, scale = max|w|."""
    v = np.asarray(vals, dtype=np.float32).reshape(-1)
    d = float(np.max(np.abs(v))) if v.size else 0.0
    if d <= 0.0:
        return np.zeros_like(v, dtype=np.int8), 0.0
    codes = np.clip(np.rint(v / d), -1, 1).astype(np.int8)
    return codes, d


def extract_trit_latent(weight: np.ndarray, group_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Return (codes int8 same shape, scales float32 with shape [..., n_groups])."""
    w = np.asarray(weight, dtype=np.float32)
    if w.shape[-1] % group_size:
        raise ValueError(f"last dim {w.shape[-1]} not divisible by group_size={group_size}")
    flat = w.reshape(-1, group_size)
    scales = np.max(np.abs(flat), axis=-1).astype(np.float32)
    scales = np.maximum(scales, 1e-8)
    codes = np.clip(np.rint(flat / scales[:, None]), -1, 1).astype(np.int8)
    return codes.reshape(w.shape), scales.reshape(*w.shape[:-1], w.shape[-1] // group_size)


# ---- PQ2_0 ----

def pack_pq2_block(codes: np.ndarray, scale: float) -> bytes:
    """codes length 128 in {-1,0,+1} → 34-byte block."""
    c = np.asarray(codes, dtype=np.int8).reshape(QK_PQ2)
    digits = (c.astype(np.int16) + 1).astype(np.uint8)  # 0,1,2
    qs = np.zeros(QK_PQ2 // 4, dtype=np.uint8)
    for i, dig in enumerate(digits):
        qs[i // 4] |= np.uint8((int(dig) & 0x3) << (2 * (i % 4)))
    return _f16_bytes(scale) + qs.tobytes()


def dequant_pq2_block(block: bytes | memoryview) -> np.ndarray:
    raw = memoryview(block)
    d = _read_f16(raw[:2])
    qs = np.frombuffer(raw[2:PQ2_BYTES], dtype=np.uint8)
    out = np.empty(QK_PQ2, dtype=np.float32)
    for i in range(QK_PQ2):
        dig = (int(qs[i // 4]) >> (2 * (i % 4))) & 0x3
        out[i] = (dig - 1) * d
    return out


def pack_pq2(weight: np.ndarray) -> bytes:
    w = np.asarray(weight, dtype=np.float32).reshape(-1)
    if w.size % QK_PQ2:
        raise ValueError("PQ2_0 requires numel % 128 == 0")
    out = bytearray()
    for i in range(0, w.size, QK_PQ2):
        codes, d = ternarize_group(w[i : i + QK_PQ2])
        out.extend(pack_pq2_block(codes, d))
    return bytes(out)


def dequant_pq2(blob: bytes, n: int) -> np.ndarray:
    if n % QK_PQ2:
        raise ValueError("n must be multiple of 128")
    nb = n // QK_PQ2
    if len(blob) != nb * PQ2_BYTES:
        raise ValueError(f"expected {nb * PQ2_BYTES} bytes, got {len(blob)}")
    out = np.empty(n, dtype=np.float32)
    for i in range(nb):
        out[i * QK_PQ2 : (i + 1) * QK_PQ2] = dequant_pq2_block(
            blob[i * PQ2_BYTES : (i + 1) * PQ2_BYTES]
        )
    return out


# ---- PTQ1_0 ----

def _rescale_base3(q: int) -> int:
    # ceil(q * 256 / 243) so decoder ((u8)(b*3^n)*3)>>8 recovers digits
    return (q * 256 + 242) // 243


def _code_f(w: float, inv_d: float) -> int:
    return int(np.clip(np.rint(w * inv_d), -1, 1)) + 1


def pack_ptq1_block(vals: np.ndarray) -> bytes:
    blk = np.asarray(vals, dtype=np.float32).reshape(QK_PTQ1)
    d = float(np.max(np.abs(blk)))
    inv = (1.0 / d) if d > 0 else 0.0
    qs = bytearray(PTQ1_QS)
    qh = bytearray(PTQ1_QH)
    base = 0
    j = 0
    for c in PTQ1_STAGES:
        while j + c <= PTQ1_QS:
            for m in range(c):
                q = 0
                for n in range(5):
                    q = q * 3 + _code_f(float(blk[base + m + n * c]), inv)
                qs[j + m] = _rescale_base3(q)
            base += 5 * c
            j += c
    for h in range(PTQ1_QH):
        q = 0
        for m in range(4):
            q = q * 3 + _code_f(float(blk[base + h + m * PTQ1_QH]), inv)
        q *= 3  # only 4 trits — shift into top digit positions
        qh[h] = _rescale_base3(q)
    return bytes(qs) + bytes(qh) + _f16_bytes(d)


def _trit_from_byte(byte: int, n: int) -> int:
    q = (byte * POW3[n]) & 0xFF
    return ((q * 3) >> 8) - 1


def dequant_ptq1_block(block: bytes | memoryview) -> np.ndarray:
    raw = memoryview(block)
    qs = raw[:PTQ1_QS]
    qh = raw[PTQ1_QS : PTQ1_QS + PTQ1_QH]
    d = _read_f16(raw[PTQ1_QS + PTQ1_QH : PTQ1_BYTES])
    out = np.empty(QK_PTQ1, dtype=np.float32)
    o = 0
    j = 0
    for c in PTQ1_STAGES:
        while j + c <= PTQ1_QS:
            for n in range(5):
                for m in range(c):
                    out[o] = _trit_from_byte(qs[j + m], n) * d
                    o += 1
            j += c
    for n in range(4):
        for h in range(PTQ1_QH):
            out[o] = _trit_from_byte(qh[h], n) * d
            o += 1
    if o != QK_PTQ1:
        raise RuntimeError(f"PTQ1 dequant wrote {o} values")
    return out


def pack_ptq1(weight: np.ndarray) -> bytes:
    w = np.asarray(weight, dtype=np.float32).reshape(-1)
    if w.size % QK_PTQ1:
        raise ValueError("PTQ1_0 requires numel % 128 == 0")
    out = bytearray()
    for i in range(0, w.size, QK_PTQ1):
        out.extend(pack_ptq1_block(w[i : i + QK_PTQ1]))
    return bytes(out)


def dequant_ptq1(blob: bytes, n: int) -> np.ndarray:
    if n % QK_PTQ1:
        raise ValueError("n must be multiple of 128")
    nb = n // QK_PTQ1
    if len(blob) != nb * PTQ1_BYTES:
        raise ValueError(f"expected {nb * PTQ1_BYTES} bytes, got {len(blob)}")
    out = np.empty(n, dtype=np.float32)
    for i in range(nb):
        out[i * QK_PTQ1 : (i + 1) * QK_PTQ1] = dequant_ptq1_block(
            blob[i * PTQ1_BYTES : (i + 1) * PTQ1_BYTES]
        )
    return out


# ---- Q2_0 (group 64) ----

def pack_q2_block(vals: np.ndarray) -> bytes:
    blk = np.asarray(vals, dtype=np.float32).reshape(QK_Q2)
    d = float(np.max(np.abs(blk)))
    inv = (1.0 / d) if d > 0 else 0.0
    # already-ternary → digits in {0,1,2}; keep clip to [-1,2] for parity with public script
    qs_digits = np.clip(np.rint(blk * inv), -1, 2).astype(np.int8) + 1
    qs = np.zeros(QK_Q2 // 4, dtype=np.uint8)
    for i, dig in enumerate(qs_digits):
        qs[i // 4] |= np.uint8((int(dig) & 0x3) << (2 * (i % 4)))
    return _f16_bytes(d) + qs.tobytes()


def dequant_q2_block(block: bytes | memoryview) -> np.ndarray:
    raw = memoryview(block)
    d = _read_f16(raw[:2])
    qs = np.frombuffer(raw[2:Q2_BYTES], dtype=np.uint8)
    out = np.empty(QK_Q2, dtype=np.float32)
    for i in range(QK_Q2):
        dig = (int(qs[i // 4]) >> (2 * (i % 4))) & 0x3
        out[i] = (dig - 1) * d
    return out


def pack_q2(weight: np.ndarray) -> bytes:
    w = np.asarray(weight, dtype=np.float32).reshape(-1)
    if w.size % QK_Q2:
        raise ValueError("Q2_0 requires numel % 64 == 0")
    out = bytearray()
    for i in range(0, w.size, QK_Q2):
        out.extend(pack_q2_block(w[i : i + QK_Q2]))
    return bytes(out)


def dequant_q2(blob: bytes, n: int) -> np.ndarray:
    if n % QK_Q2:
        raise ValueError("n must be multiple of 64")
    nb = n // QK_Q2
    if len(blob) != nb * Q2_BYTES:
        raise ValueError(f"expected {nb * Q2_BYTES} bytes, got {len(blob)}")
    out = np.empty(n, dtype=np.float32)
    for i in range(nb):
        out[i * QK_Q2 : (i + 1) * QK_Q2] = dequant_q2_block(
            blob[i * Q2_BYTES : (i + 1) * Q2_BYTES]
        )
    return out


LAYOUTS = {
    "PQ2_0": {
        "ggml_type": GGML_TYPE_PQ2_0,
        "file_type": LLAMA_FTYPE_MOSTLY_PQ2_0,
        "group": QK_PQ2,
        "block_bytes": PQ2_BYTES,
        "pack": pack_pq2,
        "dequant": dequant_pq2,
    },
    "PTQ1_0": {
        "ggml_type": GGML_TYPE_PTQ1_0,
        "file_type": LLAMA_FTYPE_MOSTLY_PTQ1_0,
        "group": QK_PTQ1,
        "block_bytes": PTQ1_BYTES,
        "pack": pack_ptq1,
        "dequant": dequant_ptq1,
    },
    "Q2_0": {
        "ggml_type": GGML_TYPE_Q2_0,
        "file_type": LLAMA_FTYPE_MOSTLY_Q2_0,
        "group": QK_Q2,
        "block_bytes": Q2_BYTES,
        "pack": pack_q2,
        "dequant": dequant_q2,
    },
}


def same_weight_parity(weight: np.ndarray, layouts: Iterable[str] = ("PQ2_0", "PTQ1_0")) -> dict:
    """Pack weight into layouts and compare dequantized values (ternary grid)."""
    w = np.asarray(weight, dtype=np.float32).reshape(-1)
    # Snap to g128 ternary first so layouts share identical latent
    codes, scales = extract_trit_latent(w.reshape(1, -1), 128)
    snapped = (codes.astype(np.float32) * np.repeat(scales, 128)).reshape(-1)
    refs = {}
    for name in layouts:
        meta = LAYOUTS[name]
        blob = meta["pack"](snapped)
        back = meta["dequant"](blob, snapped.size)
        refs[name] = back
    names = list(refs)
    ok = True
    max_abs = 0.0
    for a in names:
        for b in names:
            if a >= b:
                continue
            diff = float(np.max(np.abs(refs[a] - refs[b])))
            max_abs = max(max_abs, diff)
            if diff > 1e-3:  # fp16 scale noise tolerance
                ok = False
    return {"ok": ok, "max_abs_diff": max_abs, "layouts": names, "numel": int(snapped.size)}
