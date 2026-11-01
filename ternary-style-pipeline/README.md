# Ternary style pipeline — QAT → PQ2_0 / PTQ1_0

Reproducible workflow to **keep a teacher model’s style/ability** while adapting an unpacked **Ternary-Bonsai-2** student via public `ternary_QAT`, then packing the **same** `{−1,0,+1}` + group-128 weights as:

| Layout | Role |
| --- | --- |
| **PQ2_0** | Primary delivery (faster PP on Prism) |
| **PTQ1_0** | Same weights, denser file / Ada decode |
| **Q2_0** | Optional upstream llama.cpp (`b11555+`) |

`PQ2_0` / `PTQ1_0` / `Q2_0` are **disk layouts**, not separate training runs.

## Prerequisites

- Unpacked student: `prism-ml/Ternary-Bonsai-2-{2B|4B|9B|27B}-unpacked` (Hadamard basis).
- Teacher: your existing model (HF FP16 preferred, or OpenAI-compatible endpoint / LM Studio).
- Python 3.10+, CUDA torch of your choice, then:

```bash
pip install -r requirements.txt
# torch first if you need a specific CUDA build
pip install "ternary_QAT[peft,transformers]"
```

- Prism LM Studio backend from the parent repo (`install.ps1` / overlay).
- Official donor GGUFs for packing (preserves `prism.hadamard.*` metadata).

**Not covered by public tools:** converting arbitrary FP16 into official Hadamard Bonsai ternary. This pipeline continues QAT on an already-unpacked Bonsai checkpoint.

## Quick dry-run (no GPU / no models)

```bash
cd ternary-style-pipeline
bash scripts/run_all.sh --dry-run
# or: powershell -File scripts/run_all.ps1 -DryRun
python -m unittest tests/test_codecs.py
```

## Phases

### 0 — Baseline

Edit `configs/models.yaml` (teacher + student ids/paths).

```bash
python eval/baseline.py --backend echo --dry-run          # wiring
python eval/baseline.py --backend openai_compat           # live teacher/student servers
python eval/baseline.py --backend hf --target bonsai       # local HF student
```

Probes: `data/probes/style_probes.jsonl`, `data/probes/capability_probes.jsonl`.

### Public capability gates (anti-degradation)

Style distill can quietly hurt math / code / tools. Before and after QAT, run suites that
follow **public scoring protocols** (smoke subsets ship in-repo; enlarge or call lm-eval for prod):

| Suite | Protocol | Gate metric | Typical max abs drop |
| --- | --- | --- | --- |
| **GSM8K** | Cobbe et al. flexible-extract | accuracy | 3% |
| **MMLU** | Hendrycks letter accuracy | accuracy | 2% |
| **IFEval** | Zhou et al. strict/loose | prompt_loose_accuracy | 5% |
| **HumanEval** | Chen et al. pass@1 (exec) | pass_at_1 | 5% |
| **BFCL-lite** | Berkeley Function-Calling AST exact | ast_exact_accuracy | 5% |
| **C-Eval** (optional) | Huang et al. letter accuracy | accuracy | 3% |
| **MT-Bench lite** (optional) | Zheng et al. rubric / LLM-judge | rubric_mean | 0.15 / 10 |

```bash
# baseline (pre-QAT Bonsai)
python eval/public_benchmarks.py --backend openai_compat \
  --target baseline_bonsai --out out/eval/public_baseline_bonsai.json

# after QAT / ship
python eval/public_benchmarks.py --backend openai_compat \
  --target post_qat --out out/eval/public_post_qat.json

python eval/regression_gates.py   # fails if required suite drop > max_abs_drop
```

Thresholds: `configs/eval_gates.yaml`. For full official numbers set `policy.use_lm_eval: true`
and install [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(and optionally OpenCompass for C-Eval).

### 1 — Distill + mix

```bash
python distill_teacher.py          # → data/teacher_traces.jsonl
python mix_corpus.py               # 80% distill / 20% capability → data/train_mixed.jsonl
```

Replace probe seeds with your real chat/tool logs for production. Keep capability anchors **few and clean**.

### 2 — QAT LoRA

```bash
python train_qat.py --dry-run      # validates data/config
python train_qat.py                # swap_linear g128 + LoRA
```

Critical knobs in `configs/lora_qat.yaml`:

- `learning_rate`: **5e-4 … 1e-3** (ternary needs ~10–50× FP FT LR).
- `code_flip`: target **0.3%–2%** (~**0.7%** sweet spot). Near-zero flip ⇒ only scales moved.
- `num_train_epochs`: **1–2**, early-stop on style/tool metrics.
- Save path: `ternarize_lora_params` → `merge_and_unload` → `reternarize_merged_linears` → `out/qat_merged/`.

### 3 — Pack

```bash
python pack_gguf.py --dry-run      # synthetic + PQ2/PTQ1 parity
python pack_gguf.py                # donor_rewrite → out/gguf/
```

`donor_rewrite` loads an official Bonsai GGUF, replaces ternary tensors from the merged HF dir (or layout-converts donor tensors when HF is absent), sets `general.file_type`, optionally hardlinks `mmproj` and runs `../graft_mtp.py`.

### 4 — Ship to LM Studio

```powershell
.\scripts\ship_lms.ps1
python scripts/smoke_prism.py
python eval/compare_teacher.py
```

Load rules (already burned once):

- FA KV: **K=`q8_0`, V=`q4_0`**
- **Do not** pass `--kv-mean-center` with K=`q8_0`
- MTP: `n-max=2`, `np=1`, **`parallel=1`**

## Layout

```text
ternary-style-pipeline/
  configs/           models, distill, lora_qat, pack, eval, eval_gates, lms
  data/probes/       style + capability + public/* smoke suites
  data/samples/      example JSONL
  eval/              baseline, public_benchmarks, regression_gates, scorers/
  ternary_pipeline/  codecs (PQ2/PTQ1/Q2), gguf_io, code_flip
  train_qat.py
  distill_teacher.py
  mix_corpus.py
  pack_gguf.py
  scripts/run_all.ps1 | run_all.sh | ship_lms.ps1 | smoke_prism.py
  tests/test_codecs.py | test_scorers.py
```

## Success criteria

1. Step-0 (swap, untrained) loads and is non-gibberish on Prism (format/Hadamard OK).
2. After QAT: style closer to teacher than baseline Bonsai; **public gates pass** (GSM8K/MMLU/IFEval/HumanEval/BFCL drops within `eval_gates.yaml`); code-flip in band.
3. Same checkpoint → PQ2_0 and PTQ1_0 **greedy text match** (same-weight). Codec parity is checked in `tests/test_codecs.py` and `pack_gguf.py --dry-run`.
