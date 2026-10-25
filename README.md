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
# local_path -> existing Prism cuda folder
# url        -> zip download (default uses ghfast.top; do not use gh-proxy.net)
```

## Install

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

## Smoke test

```powershell
.\test_smoke.ps1
```

Loads `model_for_test` (default Ternary-Bonsai PQ2_0) with `-ngl 8` via the installed backend’s
`llama-server.exe`.

## CUDA note

Donor runtimes often ship **CUDA 12** vendor libs. Prism release zips may be **CUDA 13**.
If LM Studio fails to load the backend, either:

- Use a CUDA 12 Prism build, or
- Copy matching `cudart` / vendor DLLs next to `llama-server.exe`

## Rollback

Delete:

`%USERPROFILE%\.lmstudio\extensions\backends\<backend_name>-<backend_version>\`

Then pick an official runtime again in Ctrl+Shift+R.

## CLI (lmstudio-js fork)

After building the cybort `lmstudio-js` / `lms` fork:

```powershell
lms runtime register-custom --config K:\project\lmstudio-prism\config.json --prism-root K:\project\lmstudio-prism
# or:
lms runtime register-custom --path K:\project\Strata\third_party\prism-llama\cuda
```

See `K:\project\lmstudio-js\PRISM.md`.

## Related

- Prism pin used here: `prism-b10777-e3de715`
- Strata Bonsai path: `--engine bonsai` + `tools/bonsai/fetch_prism.py`
