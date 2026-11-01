#Requires -Version 5.1
<#
.SYNOPSIS
  Install PrismML llama.cpp into an LM Studio backend folder.

.DESCRIPTION
  Reads config.json (copies from config.example.json if missing).
  mode=local|url for Prism source.
  install_target=overlay (default): backup + overlay into donor_backend (shows in lms runtime ls).
  install_target=sidecar: new folder + register_index.py (needs LM Studio restart to appear).
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
$installTarget = if ($cfg.install_target) { $cfg.install_target } else { "overlay" }
$cacheDir = Join-Path $Root "cache"

function Assert-Path($path, $label) {
  if (-not (Test-Path $path)) { throw "$label not found: $path" }
}

function Overlay-Prism($destDir, $prismDir) {
  $prismOverlay = @(
    "llama-server.exe",
    "llama-server-impl.dll",
    "llama.dll",
    "llama-common.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cuda.dll",
    "ggml-rpc.dll",
    "mtmd.dll",
    "llama-kv-mean-center.exe"
  )
  foreach ($name in $prismOverlay) {
    $src = Join-Path $prismDir $name
    if (Test-Path $src) {
      Copy-Item $src (Join-Path $destDir $name) -Force
      Write-Host "  overlay $name"
    }
  }
  $cpuDonor = Join-Path $destDir "ggml-cpu.dll"
  $cpuPrism = Get-ChildItem $prismDir -Filter "ggml-cpu*.dll" | Sort-Object Length -Descending | Select-Object -First 1
  if ($cpuPrism) {
    Copy-Item $cpuPrism.FullName $cpuDonor -Force
    Write-Host "  overlay ggml-cpu.dll from $($cpuPrism.Name)"
  }
  Get-ChildItem $prismDir -Filter "ggml-cpu-*.dll" | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $destDir $_.Name) -Force
  }
}

Assert-Path $donorDir "donor_backend"

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

$outDir = $null
if ($installTarget -eq "overlay") {
  $backup = "$donorDir.bak-stock"
  if (-not (Test-Path $backup)) {
    Write-Host "[lmstudio-prism] backing up stock donor -> $backup"
    Copy-Item -Path $donorDir -Destination $backup -Recurse -Force
  }
  elseif ($Force) {
    Write-Host "[lmstudio-prism] stock backup already at $backup (kept)"
  }
  Write-Host "[lmstudio-prism] install_target=overlay into $donorDir"
  Overlay-Prism $donorDir $prismDir
  # Keep stock manifest name/version so lms runtime ls keeps listing it
  $outDir = $donorDir
  Write-Host "[lmstudio-prism] overlay done (runtime still listed as donor name/version)"
}
elseif ($installTarget -eq "sidecar") {
  $outName = "$($cfg.backend_name)-$($cfg.backend_version)"
  $outDir = Join-Path $backendsRoot $outName
  if ((Test-Path $outDir) -and -not $Force) {
    Write-Host "[lmstudio-prism] already installed: $outDir (pass -Force to rebuild)"
  }
  else {
    if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host "[lmstudio-prism] copying donor from $donorDir"
    Copy-Item -Path (Join-Path $donorDir "*") -Destination $outDir -Recurse -Force
    Overlay-Prism $outDir $prismDir
    $manifestPath = Join-Path $outDir "backend-manifest.json"
    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
    $manifest.name = $cfg.backend_name
    $manifest.version = $cfg.backend_version
    $manifest.engine_protocol_server = @{
      runtime_kind = "llama-server"
      executable_relative_path = "llama-server.exe"
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 10), $utf8NoBom)
    Write-Host "[lmstudio-prism] sidecar installed -> $outDir"
  }
  $donorManifest = Get-Content (Join-Path $donorDir "backend-manifest.json") -Raw | ConvertFrom-Json
  python (Join-Path $Root "register_index.py") `
    --backend-dir $outDir `
    --donor-name $donorManifest.name `
    --donor-version $donorManifest.version `
    --name $cfg.backend_name `
    --version $cfg.backend_version
  if ($LASTEXITCODE -ne 0) { throw "register_index.py failed" }
  Write-Host "[lmstudio-prism] note: exit LM Studio fully, then reopen, for sidecar to appear in runtime ls"
}
else {
  throw "install_target must be 'overlay' or 'sidecar', got: $installTarget"
}

Write-Host ""
Write-Host "Next:"
if ($installTarget -eq "overlay") {
  Write-Host "  Keep selecting the donor runtime in Ctrl+Shift+R (now runs Prism binaries)."
  Write-Host "  Rollback: copy from $donorDir.bak-stock back over the donor folder."
}
else {
  Write-Host "  1. Fully quit LM Studio, reopen, Ctrl+Shift+R -> '$($cfg.backend_name)@$($cfg.backend_version)'"
  Write-Host "  2. Or: lms runtime ls"
}
Write-Host "  Smoke:  .\test_smoke.ps1"
Write-Host ""
Write-Host "BACKEND_DIR=$outDir"
