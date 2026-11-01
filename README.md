# lmstudio-prism

Install **PrismML llama.cpp** (PQ2_0 / PTQ1_0 / Hadamard) as an LM Studio **runtime backend**.

The LM Studio desktop app is closed source. This project does not fork it. It builds a custom
backend folder under `~/.lmstudio/extensions/backends/` by:

1. Copying bridge files from an official CUDA donor runtime
2. Overlaying Prism `llama-server` + `ggml-*.dll`

## Config

```powershell
copy config.example.json config.json
# edit mode: "local" | "url"
# local_path -> existing Prism cuda folder (or vendor/cuda-fa after build-fa-quants.ps1)
# url        -> zip download (default uses ghfast.top; do not use gh-proxy.net)
```

## Install (stock Prism release zip)

```powershell
.\install.ps1
# rebuild:
.\install.ps1 -Force
```

`install_target` in config:

| Value | Behavior |
| --- | --- |
| **`overlay`** (default) | Backs up the donor folder to `*.bak-stock`, then overlays Prism DLLs/exe into it. Still appears as the donor in `lms runtime ls` / Ctrl+Shift+R. |
| **`sidecar`** | New backend folder + `register_index.py`. Fully quit LM Studio and reopen before it shows up. |

```powershell
lms runtime ls
# overlay: keep selecting the donor (e.g. cuda12@2.60.0) — binaries are Prism
```

## Build FA kernels for asymmetric quantized KV

Stock Prism CUDA zips only ship FlashAttention for `q4_0-q4_0` / `q8_0-q8_0` / `f16-f16`.
Mixed pairs (e.g. **K=q8_0, V=q4_0**) fall back to converting KV to f16 (slow warning in the log).

```powershell
# Needs VS Build Tools 2026 + CUDA toolkit. Default arch=89 (RTX 4080).
.\build-fa-quants.ps1
# stages Release binaries to vendor/cuda-fa

# point config local_path at vendor/cuda-fa (already in config.example.json), then:
.\install.ps1 -Force
```

Compiled FA pairs (default):

- Symmetric: `q4_0-q4_0`, `q8_0-q8_0`, `f16-f16`, `bf16-bf16`
- Asymmetric: `q8_0-q4_0`, `f16-q8_0`, `f16-q4_0`, `bf16-q8_0`, `bf16-q4_0`

Recommended LM Studio KV setting after this build: **K=q8_0, V=q4_0** (K higher precision).

## K-cache mean-center bias (for q4_0 keys)

When using `--cache-type-k q4_0`, generate a model-specific bias:

```powershell
.\make-kv-bias.ps1
# writes <model>-kv-bias.gguf next to the GGUF
```

Pass to the server (LM Studio → load → additional llama.cpp args):

```text
--kv-mean-center J:/models/prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0-kv-bias.gguf
```

Requires `--cache-type-k q4_0`. See Prism `docs/kv-mean-center.md`.

## Graft MTP head (PQ2_0 / PTQ1_0)

Official 27B GGUFs ship without an MTP block. Graft 15 `blk.64.*` tensors (same layout as
stock Qwen3.8 / community MTP models) and set `qwen35.nextn_predict_layers=1` so LM Studio
sets `supportsMtp: true`:

```powershell
# PQ2_0 (default paths). Uses a local Qwen3.8 MTP donor if present, else HF mirror.
.\graft-mtp.ps1

# PTQ1_0 after the base file is downloaded next to PQ2_0:
.\graft-mtp.ps1 -Base J:\models\prism-ml\Ternary-Bonsai-2-27B-gguf\Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

Output example: `J:\models\prism-ml\Ternary-Bonsai-2-27B-PQ2_0-MTP-gguf\Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf`
(own folder so LM Studio indexes it like other MTP models; `mmproj` / `kv-bias` are hardlinked).

Base tensors stay bit-identical; MTP weights are **not** added to `prism.hadamard.weight_names`.
Metadata: `qwen35.block_count=65`, `qwen35.nextn_predict_layers=1` → LMS `supportsMtp: true`.

After grafting, refresh **My Models** in LM Studio (or restart the app) so the indexer picks up the new folder. Then load with Draft MTP:

```powershell
lms load prism-ml/ternary-bonsai-2-27b-pq2_0-mtp-gguf/ternary-bonsai-2-27b-pq2_0-mtp.gguf `
  --speculative-draft-mtp --speculative-draft-max-tokens 2 -y
```

UI: Speculative decoding → **Draft MTP**, max draft tokens `2`, parallel slots `1`.
CLI equivalent: `--spec-type draft-mtp --spec-draft-n-max 2`.

Prism `qwen35` from ~`prism-b10743` includes the MTP Hadamard inverse; this overlay
(`prism-b10777` + FA) has it.

## Smoke test

```powershell
.\test_smoke.ps1
```

Loads `model_for_test` (default Ternary-Bonsai PQ2_0) with `-ngl 8` via the installed backend’s
`llama-server.exe`.

## CUDA note

Donor runtimes often ship **CUDA 12** vendor libs. Prism release zips / local FA builds may be **CUDA 13**.
If LM Studio fails to load the backend, either:

- Use a CUDA 12 Prism build, or
- Copy matching `cudart` / vendor DLLs next to `llama-server.exe`

`build-fa-quants.ps1` defaults to **one** CUDA arch (`89`). Rebuild with `-CudaArch "86;89;120"` if you need other GPUs.

## Rollback

Delete / restore:

`%USERPROFILE%\.lmstudio\extensions\backends\<donor>.bak-stock` → copy back over the donor folder.

Then pick an official runtime again in Ctrl+Shift+R.

## Related

- Prism pin: `prism-b10777-e3de715`
- Strata Bonsai path: `--engine bonsai` + `tools/bonsai/fetch_prism.py`
