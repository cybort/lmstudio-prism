#Requires -Version 5.1
<#
.SYNOPSIS
  Install PrismML llama.cpp into an LM Studio backend folder.

.DESCRIPTION
  Reads config.json (copies from config.example.json if missing).
  mode=local  -> use local_path
  mode=url    -> download zip into cache/ and extract
  Copies LM Studio bridge files from donor_backend, then overlays Prism runtime binaries.
#>
param(
  [string]$ConfigPath = "",
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
if (-not $ConfigPath) { $ConfigPath = Join-Path $Root "config.json" }
$Example = Join-Path $Root "config.example.json"

if (-not (Test-Path $ConfigPath)) {
  Copy-Item $Example $ConfigPath
  Write-Host "[lmstudio-prism] created $ConfigPath from example"
}

$cfg = Get-Content $ConfigPath -Raw | ConvertFrom-Json
$backendsRoot = Join-Path $env:USERPROFILE ".lmstudio\extensions\backends"
$donorDir = Join-Path $backendsRoot $cfg.donor_backend
$outName = "$($cfg.backend_name)-$($cfg.backend_version)"
$outDir = Join-Path $backendsRoot $outName
$cacheDir = Join-Path $Root "cache"

function Assert-Path($path, $label) {
  if (-not (Test-Path $path)) { throw "$label not found: $path" }
}

Assert-Path $donorDir "donor_backend"

# Resolve Prism directory
$prismDir = $null
if ($cfg.mode -eq "local") {
  Assert-Path $cfg.local_path "local_path"
  $prismDir = (Resolve-Path $cfg.local_path).Path
  Write-Host "[lmstudio-prism] mode=local -> $prismDir"
}
elseif ($cfg.mode -eq "url") {
  if (-not $cfg.url) { throw "mode=url requires url" }
  New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
  $zipName = [IO.Path]::GetFileName(($cfg.url -split '\?')[0])
  if (-not $zipName.EndsWith(".zip")) { $zipName = "prism-cuda.zip" }
  $zipPath = Join-Path $cacheDir $zipName
  $extractDir = Join-Path $cacheDir ([IO.Path]::GetFileNameWithoutExtension($zipName))
  if (-not (Test-Path $zipPath) -or $Force) {
    Write-Host "[lmstudio-prism] downloading $($cfg.url)"
    curl.exe -L --retry 5 --retry-delay 3 --connect-timeout 60 --progress-bar -o $zipPath $cfg.url
    if ($LASTEXITCODE -ne 0) { throw "download failed exit=$LASTEXITCODE" }
  }
  if ((Get-Item $zipPath).Length -lt 1MB) {
    throw "download looks too small ($((Get-Item $zipPath).Length) bytes); check URL/proxy"
  }
  if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
  Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
  $server = Get-ChildItem -Recurse $extractDir -Filter "llama-server.exe" | Select-Object -First 1
  if (-not $server) { throw "llama-server.exe not found in extracted zip" }
  $prismDir = $server.Directory.FullName
  Write-Host "[lmstudio-prism] mode=url -> $prismDir"
}
else {
  throw "mode must be 'local' or 'url', got: $($cfg.mode)"
}

Assert-Path (Join-Path $prismDir "llama-server.exe") "Prism llama-server.exe"
Assert-Path (Join-Path $prismDir "ggml-cuda.dll") "Prism ggml-cuda.dll"

if ((Test-Path $outDir) -and -not $Force) {
  Write-Host "[lmstudio-prism] already installed: $outDir (pass -Force to rebuild)"
}
else {
  if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null

  # 1) Donor: LM Studio bridge + runtime scaffolding
  Write-Host "[lmstudio-prism] copying donor from $donorDir"
  Copy-Item -Path (Join-Path $donorDir "*") -Destination $outDir -Recurse -Force

  # 2) Overlay Prism runtime binaries (keep *.node / llm_engine* / lmstudiocore)
  $prismOverlay = @(
    "llama-server.exe",
    "llama-server-impl.dll",
    "llama.dll",
    "llama-common.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cuda.dll",
    "ggml-rpc.dll",
    "mtmd.dll"
  )
  foreach ($name in $prismOverlay) {
    $src = Join-Path $prismDir $name
    if (Test-Path $src) {
      Copy-Item $src (Join-Path $outDir $name) -Force
      Write-Host "  overlay $name"
    }
  }
  # Prism ships multiple ggml-cpu-*.dll; LM Studio expects ggml-cpu.dll
  $cpuDonor = Join-Path $outDir "ggml-cpu.dll"
  $cpuPrism = Get-ChildItem $prismDir -Filter "ggml-cpu*.dll" | Sort-Object Length -Descending | Select-Object -First 1
  if ($cpuPrism) {
    # Prefer a single ggml-cpu.dll name for LM Studio; keep largest Prism CPU backend as ggml-cpu.dll
    # but do not delete donor if Prism only has arch-specific names without a generic one.
    if ($cpuPrism.Name -eq "ggml-cpu.dll") {
      Copy-Item $cpuPrism.FullName $cpuDonor -Force
      Write-Host "  overlay ggml-cpu.dll"
    }
    else {
      Copy-Item $cpuPrism.FullName (Join-Path $outDir $cpuPrism.Name) -Force
      Copy-Item $cpuPrism.FullName $cpuDonor -Force
      Write-Host "  overlay ggml-cpu.dll from $($cpuPrism.Name)"
    }
  }
  # Also copy remaining Prism ggml-cpu-* variants for Prism's own CPU dispatch
  Get-ChildItem $prismDir -Filter "ggml-cpu-*.dll" | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $outDir $_.Name) -Force
  }

  # 3) Manifest: rename engine, keep protocol + vendor from donor
  $manifestPath = Join-Path $outDir "backend-manifest.json"
  $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
  $manifest.name = $cfg.backend_name
  $manifest.version = $cfg.backend_version
  $manifest.engine_protocol_server = @{
    runtime_kind = "llama-server"
    executable_relative_path = "llama-server.exe"
  }
  $manifest | ConvertTo-Json -Depth 10 | Set-Content $manifestPath -Encoding UTF8

  # 4) display-data: mark as Prism
  $displayPath = Join-Path $outDir "display-data.json"
  if (Test-Path $displayPath) {
    try {
      $display = Get-Content $displayPath -Raw | ConvertFrom-Json
      # display-data is [["en",{...}], ...]
      foreach ($entry in $display) {
        if ($entry -is [System.Array] -or $entry.Count -ge 2) {
          $meta = $entry[1]
          if ($meta.displayName) {
            $meta.displayName = "Prism CUDA llama.cpp (Windows)"
            $meta.description = "PrismML llama.cpp with PQ2_0 / PTQ1_0 (overlay on LM Studio donor bridge)"
          }
        }
      }
      $display | ConvertTo-Json -Depth 20 -Compress | Set-Content $displayPath -Encoding UTF8
    }
    catch {
      Write-Host "[lmstudio-prism] warning: could not rewrite display-data.json: $_"
    }
  }

  Write-Host "[lmstudio-prism] installed -> $outDir"
}

Write-Host ""
Write-Host "Next:"
Write-Host "  1. Ctrl+Shift+R in LM Studio -> select '$($cfg.backend_name)@$($cfg.backend_version)'"
Write-Host "  2. Or: lms runtime ls"
Write-Host "  3. Smoke:  .\test_smoke.ps1"
Write-Host ""
Write-Host "BACKEND_DIR=$outDir"
