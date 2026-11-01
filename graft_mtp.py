#!/usr/bin/env python3
"""
Graft Qwen3.8-27B MTP head (blk.64.*) into a PrismML Ternary-Bonsai-2-27B GGUF.

Based on decent-jawfish/bonsai-2-27b-mtp graft_mtp.py:
  - Sets qwen35.block_count=65 and qwen35.nextn_predict_layers=1
  - Copies 15 blk.64.* tensors verbatim (NOT into prism.hadamard.weight_names)
  - Leaves base ternary weights byte-identical

Sources (first that works):
  1) --mtp-src local GGUF that already has blk.64.*
  2) --mtp-url remote GGUF (HTTP Range)
  3) default: ProCreations Bonsai MTP (trained head), via HF_ENDPOINT / hf-mirror
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import urllib.request


DEFAULT_MTP_URL = (
    "ProCreations/Ternary-Bonsai-2-27B-MTP/"
    "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"
)


class Rd:
    def __init__(self, b: bytes):
        self.b = b
        self.o = 0

    def raw(self, n: int) -> bytes:
        if self.o + n > len(self.b):
            raise EOFError("header buffer too small")
        v = self.b[self.o : self.o + n]
        self.o += n
        return v

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def st(self) -> str:
        return self.raw(self.u64()).decode("utf-8")


def rd_val(r: Rd, t: int):
    if t == 8:
        return (t, r.st())
    if t in (0, 1, 7):
        return (t, r.raw(1)[0])
    if t in (2, 3):
        return (t, struct.unpack("<H", r.raw(2))[0])
    if t in (4, 5):
        return (t, struct.unpack("<i", r.raw(4))[0])
    if t == 6:
        return (t, struct.unpack("<f", r.raw(4))[0])
    if t in (10, 11):
        return (t, struct.unpack("<q", r.raw(8))[0])
    if t == 12:
        return (t, struct.unpack("<d", r.raw(8))[0])
    if t == 9:
        et = r.u32()
        n = r.u64()
        vals = [rd_val(r, et)[1] for _ in range(n)]
        return (t, (et, vals))
    raise ValueError(f"unknown gguf value type {t}")


def wr_str(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def wr_val(f, t: int, v):
    f.write(struct.pack("<I", t))
    _wr_raw(f, t, v)


def _wr_raw(f, t: int, v):
    if t == 8:
        wr_str(f, v)
    elif t in (0, 1, 7):
        f.write(struct.pack("<B", v))
    elif t in (2, 3):
        f.write(struct.pack("<H", v))
    elif t in (4, 5):
        f.write(struct.pack("<i", v))
    elif t == 6:
        f.write(struct.pack("<f", v))
    elif t in (10, 11):
        f.write(struct.pack("<q", v))
    elif t == 12:
        f.write(struct.pack("<d", v))
    elif t == 9:
        et, vals = v
        f.write(struct.pack("<I", et))
        f.write(struct.pack("<Q", len(vals)))
        for x in vals:
            _wr_raw(f, et, x)
    else:
        raise ValueError(t)


def parse(buf: bytes):
    r = Rd(buf)
    assert r.raw(4) == b"GGUF", "not a GGUF"
    ver = r.u32()
    nt = r.u64()
    nk = r.u64()
    kv = []
    for _ in range(nk):
        k = r.st()
        t = r.u32()
        _, v = rd_val(r, t)
        kv.append((k, t, v))
    tl = []
    for _ in range(nt):
        nm = r.st()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ty = r.u32()
        off = r.u64()
        tl.append([nm, dims, ty, off])
    return ver, kv, tl, r.o


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def sizes_from_offsets(tl, data_bytes_total: int):
    order = sorted(range(len(tl)), key=lambda i: tl[i][3])
    out = {}
    for j, i in enumerate(order):
        off = tl[i][3]
        nxt = tl[order[j + 1]][3] if j + 1 < len(order) else data_bytes_total
        out[tl[i][0]] = nxt - off
    return out


def resolve_hf_url(repo_file: str) -> str:
    """repo_file = 'org/repo/file.gguf' or full https URL."""
    if repo_file.startswith("http://") or repo_file.startswith("https://"):
        return repo_file
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    # allow 'org/repo/path/file'
    parts = repo_file.split("/")
    if len(parts) < 3:
        raise ValueError(f"bad --mtp-url value: {repo_file}")
    org, repo, *rest = parts
    return f"{endpoint}/{org}/{repo}/resolve/main/{'/'.join(rest)}"


def http_range(url: str, lo: int, hi: int, retries: int = 8) -> bytes:
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                return resp.read()
        except Exception as e:
            last = e
            print(f"    retry {a + 1}: {e}", flush=True)
    raise RuntimeError(last)


def http_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as r:
        cl = r.headers.get("Content-Length") or r.headers.get("content-length")
        if not cl:
            raise RuntimeError(f"no Content-Length for {url}")
        return int(cl)


def load_header_local(path: str, nbytes: int = 48 * 1024 * 1024):
    with open(path, "rb") as f:
        return f.read(nbytes)


def load_header_remote(url: str, nbytes: int = 48 * 1024 * 1024):
    return http_range(url, 0, nbytes - 1)


def main():
    ap = argparse.ArgumentParser(description="Graft MTP head into Bonsai GGUF")
    ap.add_argument("base", help="Bonsai PQ2_0 / PTQ1_0 GGUF (no MTP)")
    ap.add_argument("out", help="Output GGUF path")
    ap.add_argument(
        "--mtp-src",
        help="Local donor GGUF that already contains blk.64.*",
    )
    ap.add_argument(
        "--mtp-url",
        default=DEFAULT_MTP_URL,
        help="Remote donor (HF path org/repo/file or full URL)",
    )
    ap.add_argument(
        "--donor-name",
        default="",
        help="Optional graft.donor.name metadata",
    )
    args = ap.parse_args()

    base = args.base
    out = args.out
    if not os.path.isfile(base):
        print(f"base not found: {base}", file=sys.stderr)
        sys.exit(2)
    if os.path.exists(out):
        print(f"refusing to overwrite existing: {out}", file=sys.stderr)
        sys.exit(2)

    print("[1/6] parsing Bonsai header", flush=True)
    bver, bkv, btl, bhdr_end = parse(load_header_local(base))
    # already has MTP?
    if any(t[0].startswith("blk.64.") for t in btl):
        print("base already has blk.64.*; nothing to do", file=sys.stderr)
        sys.exit(1)
    balign = next((v for k, t, v in bkv if k == "general.alignment"), 32)
    bdata_start = align_up(bhdr_end, balign)
    bfile = os.path.getsize(base)
    bsz = sizes_from_offsets(btl, bfile - bdata_start)
    print(
        f"      tensors={len(btl)} kv={len(bkv)} data_start={bdata_start} file={bfile}",
        flush=True,
    )

    local_mtp = args.mtp_src
    mtp_url = None
    donor_label = args.donor_name

    if local_mtp:
        print(f"[2/6] parsing local MTP donor: {local_mtp}", flush=True)
        uver, ukv, utl, uhdr_end = parse(load_header_local(local_mtp))
        ualign = next((v for k, t, v in ukv if k == "general.alignment"), 32)
        udata_start = align_up(uhdr_end, ualign)
        ufile = os.path.getsize(local_mtp)
        usz = sizes_from_offsets(utl, ufile - udata_start)
        if not donor_label:
            donor_label = os.path.basename(local_mtp)
    else:
        mtp_url = resolve_hf_url(args.mtp_url)
        print(f"[2/6] parsing remote MTP donor: {mtp_url}", flush=True)
        uver, ukv, utl, uhdr_end = parse(load_header_remote(mtp_url))
        ualign = next((v for k, t, v in ukv if k == "general.alignment"), 32)
        udata_start = align_up(uhdr_end, ualign)
        ufile = http_size(mtp_url)
        usz = sizes_from_offsets(utl, ufile - udata_start)
        if not donor_label:
            donor_label = args.mtp_url

    mtp = [t for t in utl if t[0].startswith("blk.64.")]
    # Prefer lean 15-tensor head; ignore optional nextn.embed_tokens if present
    # when counting — but still copy ALL blk.64.* (incl. embed_tokens if donor has it).
    mtp.sort(key=lambda t: t[3])
    core = [t for t in mtp if t[0] != "blk.64.nextn.embed_tokens.weight"]
    print(
        f"      found {len(mtp)} blk.64 tensors "
        f"({len(core)} core + {len(mtp) - len(core)} extra), "
        f"{sum(usz[t[0]] for t in mtp) / 1e9:.2f} GB",
        flush=True,
    )
    if len(core) != 15:
        print(f"expected 15 core MTP tensors, got {len(core)}", file=sys.stderr)
        sys.exit(1)

    print("[3/6] building merged metadata", flush=True)
    newkv = []
    seen_nextn = False
    for k, t, v in bkv:
        if k == "qwen35.block_count":
            print(f"      block_count {v} -> 65", flush=True)
            v = 65
        if k == "qwen35.nextn_predict_layers":
            v = 1
            seen_nextn = True
        if k == "general.name" and isinstance(v, str) and "MTP" not in v:
            v = v if v and v != "Hf" else "Ternary Bonsai 2 27B"
            if not v.endswith(" MTP"):
                v = f"{v} MTP"
        newkv.append((k, t, v))
    if not seen_nextn:
        idx = next(i for i, (k, _, _) in enumerate(newkv) if k == "qwen35.block_count")
        newkv.insert(idx + 1, ("qwen35.nextn_predict_layers", 4, 1))
        print("      added qwen35.nextn_predict_layers = 1", flush=True)

    # provenance (string type 8 / uint32 type 4)
    newkv.append(("graft.donor.name", 8, str(donor_label)))
    newkv.append(("graft.head_blocks", 4, 1))
    newkv.append(("graft.head_tensor_count", 4, len(mtp)))
    newkv.append(("graft.tool", 8, "lmstudio-prism/graft_mtp.py"))

    newtl = [[nm, dims, ty, 0] for nm, dims, ty, off in btl]
    for nm, dims, ty, off in mtp:
        newtl.append([nm, dims, ty, 0])

    run = 0
    plan = []
    for i, (nm, dims, ty, _) in enumerate(newtl):
        if i < len(btl):
            n = bsz[nm]
            src = ("base", btl[i][3])
        else:
            n = usz[nm]
            src = ("mtp", next(t[3] for t in mtp if t[0] == nm))
        run = align_up(run, balign)
        newtl[i][3] = run
        plan.append((nm, src[0], src[1], n, run))
        run += n
    total_data = run
    print(f"      total tensors={len(newtl)} data={total_data / 1e9:.2f} GB", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    print(f"[4/6] writing {out}", flush=True)
    with open(out, "wb") as o:
        o.write(b"GGUF")
        o.write(struct.pack("<I", bver))
        o.write(struct.pack("<Q", len(newtl)))
        o.write(struct.pack("<Q", len(newkv)))
        for k, t, v in newkv:
            wr_str(o, k)
            wr_val(o, t, v)
        for nm, dims, ty, off in newtl:
            wr_str(o, nm)
            o.write(struct.pack("<I", len(dims)))
            for d in dims:
                o.write(struct.pack("<Q", d))
            o.write(struct.pack("<I", ty))
            o.write(struct.pack("<Q", off))
        hdr_end = o.tell()
        data_start = align_up(hdr_end, balign)
        o.write(b"\x00" * (data_start - hdr_end))
        print(f"      data_start={data_start}", flush=True)

        ch = 32 * 1024 * 1024
        with open(base, "rb") as bf:
            mtp_f = open(local_mtp, "rb") if local_mtp else None
            try:
                for idx, (nm, kind, soff, n, noff) in enumerate(plan):
                    want = data_start + noff
                    cur = o.tell()
                    if cur < want:
                        o.write(b"\x00" * (want - cur))
                    assert o.tell() == want, f"offset drift at {nm}"
                    if kind == "base":
                        bf.seek(bdata_start + soff)
                        left = n
                        while left:
                            c = bf.read(min(ch, left))
                            if not c:
                                raise IOError("short read from base")
                            o.write(c)
                            left -= len(c)
                    elif local_mtp:
                        mtp_f.seek(udata_start + soff)
                        left = n
                        while left:
                            c = mtp_f.read(min(ch, left))
                            if not c:
                                raise IOError(f"short read from MTP donor at {nm}")
                            o.write(c)
                            left -= len(c)
                        print(f"      + {nm}  {n / 1e6:.1f} MB", flush=True)
                    else:
                        lo = udata_start + soff
                        left = n
                        pos = 0
                        while left:
                            c = min(ch, left)
                            o.write(http_range(mtp_url, lo + pos, lo + pos + c - 1))
                            pos += c
                            left -= c
                        print(f"      + {nm}  {n / 1e6:.1f} MB", flush=True)
                    if idx % 200 == 0:
                        print(f"      .. {idx}/{len(plan)}", flush=True)
            finally:
                if mtp_f:
                    mtp_f.close()

    print("[5/6] verifying", flush=True)
    with open(out, "rb") as f:
        vh = f.read(48 * 1024 * 1024)
    vver, vkv, vtl, vend = parse(vh)
    vdata = align_up(vend, balign)
    vsize = os.path.getsize(out)
    names = {t[0] for t in vtl}
    bc = next(v for k, t, v in vkv if k == "qwen35.block_count")
    nx = next(v for k, t, v in vkv if k == "qwen35.nextn_predict_layers")
    n64 = sum(1 for n in names if n.startswith("blk.64."))
    print(f"      tensors={len(vtl)} (base {len(btl)} + {len(mtp)} = {len(btl) + len(mtp)})")
    print(f"      block_count={bc}  nextn_predict_layers={nx}")
    print(f"      blk.64 present: {n64}/{len(mtp)}")
    print(f"      file size={vsize / 1e9:.2f} GB  data_start={vdata}")
    ok = (
        len(vtl) == len(btl) + len(mtp)
        and bc == 65
        and nx == 1
        and n64 == len(mtp)
        and vsize >= vdata + total_data
    )
    print("[6/6]", "OK" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
