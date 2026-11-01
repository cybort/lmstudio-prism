#!/usr/bin/env python3
"""Unit tests for public scoring protocols."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.scorers.bfcl_lite import score_bfcl_lite  # noqa: E402
from eval.scorers.gsm8k import extract_flexible, score_gsm8k  # noqa: E402
from eval.scorers.humaneval import score_humaneval  # noqa: E402
from eval.scorers.ifeval import score_ifeval  # noqa: E402
from eval.scorers.mmlu import extract_letter, score_mmlu_letter  # noqa: E402


class ScorerTests(unittest.TestCase):
    def test_gsm8k_flexible(self):
        self.assertEqual(extract_flexible("... #### 72"), "72")
        self.assertEqual(extract_flexible("Answer is 10."), "10")
        r = score_gsm8k(
            [
                {"id": "1", "gold": "72", "output": "#### 72"},
                {"id": "2", "gold": "10", "output": "I think 11"},
            ]
        )
        self.assertEqual(r["correct"], 1)
        self.assertAlmostEqual(r["accuracy"], 0.5)

    def test_mmlu_letter(self):
        self.assertEqual(extract_letter("The answer is B."), "B")
        r = score_mmlu_letter(
            [
                {"id": "1", "gold": "B", "output": "Answer: B"},
                {"id": "2", "gold": "C", "output": "I pick A"},
            ]
        )
        self.assertEqual(r["correct"], 1)

    def test_ifeval_json_and_bullets(self):
        r = score_ifeval(
            [
                {
                    "id": "1",
                    "instructions": [{"kind": "json_object"}],
                    "output": '{"city": "x", "country": "y"}',
                },
                {
                    "id": "2",
                    "instructions": [{"kind": "bullet_count", "n": 3}],
                    "output": "- a\n- b\n- c",
                },
            ]
        )
        self.assertEqual(r["prompt_strict_accuracy"], 1.0)

    def test_humaneval_exec(self):
        r = score_humaneval(
            [
                {
                    "id": "1",
                    "entry_point": "add",
                    "test": "assert add(1, 2) == 3\n",
                    "output": "```python\ndef add(a, b):\n    return a + b\n```",
                }
            ],
            execute=True,
        )
        self.assertEqual(r["pass_at_1"], 1.0)

    def test_bfcl_exact(self):
        r = score_bfcl_lite(
            [
                {
                    "id": "1",
                    "gold": {"name": "search", "arguments": {"q": "x"}},
                    "output": '{"name": "search", "arguments": {"q": "x"}}',
                },
                {
                    "id": "2",
                    "gold": {"name": "search", "arguments": {"q": "x"}},
                    "output": '{"name": "search", "arguments": {"q": "y"}}',
                },
            ]
        )
        self.assertAlmostEqual(r["ast_exact_accuracy"], 0.5)
        self.assertAlmostEqual(r["name_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
