# Graft MTP head (blk.64.*) into Bonsai PQ2_0 / PTQ1_0 GGUF for Prism + LM Studio.
# Usage:
#   .\graft-mtp.ps1
#   .\graft-mtp.ps1 -Base J:\models\...\Ternary-Bonsai-2-27B-PTQ1_0.gguf
#   .\graft-mtp.ps1 -MtpSrc J:\models\...\some-Qwen3.8-MTP.gguf

param(
    [string]$Base = "J:\models\prism-ml\Ternary-Bonsai-2-27B-gguf\Ternary-Bonsai-2-27B-PQ2_0.gguf",
    [string]$Out = "",
    [string]$MtpSrc = "",
    [string]$MtpUrl = "ProCreations/Ternary-Bonsai-2-27B-MTP/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf",
    [string]$HfEndpoint = $(if ($env:HF_ENDPOINT) { $env:HF_ENDPOINT } else { "https://hf-mirror.com" })
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root "graft_mtp.py"

if (-not (Test-Path $Base)) { throw "Base GGUF not found: $Base" }

$stem = [IO.Path]::GetFileNameWithoutExtension($Base)
$dir = Split-Path -Parent $Base
if (-not $Out) {
    if ($stem -match "-MTP$") { throw "Base already looks like MTP: $stem" }
    # Own folder (…-MTP-gguf) so LM Studio indexes like other MTP models
    $outDir = Join-Path (Split-Path -Parent $dir) ($stem + "-MTP-gguf")
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $Out = Join-Path $outDir ($stem + "-MTP.gguf")
    # Hardlink mmproj / kv-bias when present next to the base
    foreach ($name in @(
        "mmproj-Ternary-Bonsai-2-27B-Q8_0.gguf",
        "mmproj-Ternary-Bonsai-2-27B-BF16.gguf",
        ($stem + "-kv-bias.gguf")
    )) {
        $src = Join-Path $dir $name
        $dst = Join-Path $outDir $name
        if ((Test-Path $src) -and -not (Test-Path $dst)) {
            cmd /c mklink /H "$dst" "$src" | Out-Null
        }
    }
}

# Prefer a local Qwen3.8 MTP donor when present (same emb/ff dims as Bonsai 27B).
if (-not $MtpSrc) {
    $candidates = @(
        "J:\models\DavidAU\Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF\Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ2_M.gguf",
        "J:\models\DavidAU\Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF\Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $MtpSrc = $c; break }
    }
}

$env:HF_ENDPOINT = $HfEndpoint
$argv = @($Py, $Base, $Out, "--mtp-url", $MtpUrl)
if ($MtpSrc) {
    Write-Host "Using local MTP donor: $MtpSrc"
    $argv += @("--mtp-src", $MtpSrc)
} else {
    Write-Host "Using remote MTP donor via $HfEndpoint : $MtpUrl"
}

Write-Host "Base: $Base"
Write-Host "Out : $Out"
& python @argv
if ($LASTEXITCODE -ne 0) { throw "graft_mtp.py failed ($LASTEXITCODE)" }

Write-Host ""
Write-Host "Done. LM Studio should pick up nextn_predict_layers=1 on rescan."
Write-Host "Load with Draft MTP, e.g.:"
Write-Host "  lms load <model-key> --speculative-draft-mtp --speculative-draft-max-tokens 2 -y"
