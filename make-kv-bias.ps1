#Requires -Version 5.1
<#
.SYNOPSIS
  Generate a K-cache mean-centering bias GGUF for Bonsai (requires --cache-type-k q4_0).

.DESCRIPTION
  Uses llama-kv-mean-center from config local_path (or vendor/cuda-fa).
  Writes <model>-kv-bias.gguf next to the model by default.
  Pass to llama-server as --kv-mean-center <path> (LM Studio: additional args).
#>
param(
  [string]$ConfigPath = "",
  [string]$Model = "",
  [string]$OutBias = "",
  [string]$CalibText = "",
  [int]$Ctx = 512,
  [int]$Chunks = 8,
  [int]$Ngl = 99
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
if (-not $ConfigPath) { $ConfigPath = Join-Path $Root "config.json" }
if (-not (Test-Path $ConfigPath)) {
  Copy-Item (Join-Path $Root "config.example.json") $ConfigPath
}
$cfg = Get-Content $ConfigPath -Raw | ConvertFrom-Json

$prismDir = $cfg.local_path
if (-not $prismDir -or -not (Test-Path $prismDir)) {
  $prismDir = Join-Path $Root "vendor\cuda-fa"
}
$exe = Join-Path $prismDir "llama-kv-mean-center.exe"
if (-not (Test-Path $exe)) { throw "llama-kv-mean-center.exe not found in $prismDir (run build-fa-quants.ps1)" }

if (-not $Model) { $Model = $cfg.model_for_test }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }

if (-not $OutBias) {
  $dir = Split-Path $Model -Parent
  $base = [IO.Path]::GetFileNameWithoutExtension($Model)
  $OutBias = Join-Path $dir ($base + "-kv-bias.gguf")
}

if (-not $CalibText) {
  $CalibText = Join-Path $Root "vendor\calib-kv-mean.txt"
}
if (-not (Test-Path $CalibText)) {
  New-Item -ItemType Directory -Force -Path (Split-Path $CalibText) | Out-Null
  $seed = "Local LLM inference depends on weight footprint, KV cache, GPU offload, flash attention, and batching. Mean-centering the key cache before q4_0 packing reduces channel bias. Hybrid attention keeps most layers linear. "
  $sb = New-Object System.Text.StringBuilder
  for ($i = 1; $i -le 40; $i++) {
    [void]$sb.Append("Paragraph ").Append($i).Append(". ").Append($seed).Append("`n`n")
  }
  [IO.File]::WriteAllText($CalibText, $sb.ToString())
}

Write-Host "[kv-bias] exe=$exe"
Write-Host "[kv-bias] model=$Model"
Write-Host "[kv-bias] out=$OutBias"
& $exe -m $Model -f $CalibText -o $OutBias -c $Ctx --chunks $Chunks -ngl $Ngl -fa on
if ($LASTEXITCODE -ne 0) { throw "llama-kv-mean-center failed exit=$LASTEXITCODE" }
Write-Host "[kv-bias] wrote $OutBias ($((Get-Item $OutBias).Length) bytes)"
Write-Host "[kv-bias] use with: --cache-type-k q4_0 --kv-mean-center `"$OutBias`""
