#!/usr/bin/env bash
# End-to-end ternary style pipeline (dry-run friendly).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=1
  shift
fi

py() { command -v python >/dev/null && python "$@" || python3 "$@"; }

echo "== Phase 0: baseline =="
if [[ "$DRY" == 1 ]]; then
  py eval/baseline.py --dry-run --backend echo
else
  py eval/baseline.py --backend "${BASELINE_BACKEND:-openai_compat}"
fi

echo "== Phase 1: distill + mix =="
if [[ "$DRY" == 1 ]]; then
  py distill_teacher.py --dry-run
  py mix_corpus.py --dry-run
else
  py distill_teacher.py
  py mix_corpus.py
fi

echo "== Phase 2: QAT LoRA =="
if [[ "$DRY" == 1 ]]; then
  py train_qat.py --dry-run
else
  py train_qat.py
fi

echo "== Phase 3: pack GGUF =="
if [[ "$DRY" == 1 ]]; then
  py pack_gguf.py --dry-run
else
  py pack_gguf.py
fi

echo "== Phase 4: smoke + regression =="
py scripts/smoke_prism.py --dry-run
# copy baseline bonsai as post placeholder when dry
mkdir -p out/eval
if [[ -f out/eval/baseline_bonsai.json && ! -f out/eval/post_qat.json ]]; then
  cp out/eval/baseline_bonsai.json out/eval/post_qat.json
fi
py eval/compare_teacher.py || true

echo "done. See out/ and README.md"
