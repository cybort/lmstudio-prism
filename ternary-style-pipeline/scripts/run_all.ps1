<#
.SYNOPSIS
  End-to-end Ternary QAT → PQ2_0/PTQ1_0 pipeline runner.
#>
param(
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$dry = @()
if ($DryRun) { $dry = @("--dry-run") }

Write-Host "== Phase 0: baseline =="
if ($DryRun) {
  python eval/baseline.py --dry-run --backend echo
  python eval/public_benchmarks.py --dry-run --target baseline_bonsai --out out/eval/public_baseline_bonsai.json
} else {
  python eval/baseline.py --backend openai_compat
  python eval/public_benchmarks.py --backend openai_compat --target baseline_bonsai --out out/eval/public_baseline_bonsai.json
}

Write-Host "== Phase 1: distill + mix =="
python distill_teacher.py @dry
python mix_corpus.py @dry

Write-Host "== Phase 2: QAT LoRA =="
python train_qat.py @dry

Write-Host "== Phase 3: pack GGUF =="
python pack_gguf.py @dry

Write-Host "== Phase 4: LMS ship hints + regression =="
python scripts/smoke_prism.py --dry-run
New-Item -ItemType Directory -Force -Path out/eval | Out-Null
if ((Test-Path out/eval/baseline_bonsai.json) -and -not (Test-Path out/eval/post_qat.json)) {
  Copy-Item out/eval/baseline_bonsai.json out/eval/post_qat.json
}
if ($DryRun) {
  python eval/public_benchmarks.py --dry-run --target post_qat --out out/eval/public_post_qat.json
} else {
  python eval/public_benchmarks.py --backend openai_compat --target post_qat --out out/eval/public_post_qat.json
}
python eval/regression_gates.py
python eval/compare_teacher.py
if ($DryRun) {
  Write-Host "Dry-run complete. For real ship: .\\scripts\\ship_lms.ps1"
} else {
  .\scripts\ship_lms.ps1
}

Write-Host "done."
