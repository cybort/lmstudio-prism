"""Publicly recognized scoring protocols for capability regression gates."""

from .gsm8k import score_gsm8k
from .mmlu import score_mmlu_letter
from .ifeval import score_ifeval
from .humaneval import score_humaneval
from .bfcl_lite import score_bfcl_lite
from .ceval import score_ceval_letter
from .mt_bench_lite import score_mt_bench_lite

SCORERS = {
    "gsm8k": score_gsm8k,
    "mmlu": score_mmlu_letter,
    "ifeval": score_ifeval,
    "humaneval": score_humaneval,
    "bfcl_lite": score_bfcl_lite,
    "ceval": score_ceval_letter,
    "mt_bench_lite": score_mt_bench_lite,
}

__all__ = ["SCORERS"]
