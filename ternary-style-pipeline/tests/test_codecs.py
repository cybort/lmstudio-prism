#!/usr/bin/env python3
"""Round-trip tests for PQ2_0 / PTQ1_0 / Q2_0 codecs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ternary_pipeline.codecs import (  # noqa: E402
    dequant_pq2,
    dequant_ptq1,
    dequant_q2,
    pack_pq2,
    pack_ptq1,
    pack_q2,
    same_weight_parity,
)


def _ternary_rows(n_groups: int, group: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.05, 2.0, size=(n_groups,)).astype(np.float32)
    codes = rng.integers(-1, 2, size=(n_groups, group), dtype=np.int8)
    return (codes.astype(np.float32) * scales[:, None]).reshape(-1)


class CodecTests(unittest.TestCase):
    def test_pq2_roundtrip(self):
        w = _ternary_rows(5, 128, seed=1)
        blob = pack_pq2(w)
        self.assertEqual(len(blob), 5 * 34)
        back = dequant_pq2(blob, w.size)
        np.testing.assert_allclose(back, w, atol=1e-3, rtol=1e-3)

    def test_ptq1_roundtrip(self):
        w = _ternary_rows(5, 128, seed=2)
        blob = pack_ptq1(w)
        self.assertEqual(len(blob), 5 * 28)
        back = dequant_ptq1(blob, w.size)
        np.testing.assert_allclose(back, w, atol=1e-3, rtol=1e-3)

    def test_q2_roundtrip_ternary(self):
        # Q2 uses group 64; build ternary at 64
        w = _ternary_rows(8, 64, seed=3)
        blob = pack_q2(w)
        self.assertEqual(len(blob), 8 * 18)
        back = dequant_q2(blob, w.size)
        np.testing.assert_allclose(back, w, atol=1e-3, rtol=1e-3)

    def test_pq2_ptq1_parity(self):
        w = _ternary_rows(3, 128, seed=4)
        report = same_weight_parity(w, ("PQ2_0", "PTQ1_0"))
        self.assertTrue(report["ok"], report)


if __name__ == "__main__":
    unittest.main()
