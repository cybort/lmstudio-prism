<#
.SYNOPSIS
  Phase 4 — copy packed GGUFs into LM Studio models dir and write safe default load config.

.NOTES
  Avoid --kv-mean-center when K cache is q8_0 (Prism crash). Prefer k=q8_0 v=q4_0 FA pair.
#>
param(
  [string]$Config = "configs/lms.yaml",
  [string]$GgufDir = "",
  [string]$ShipRoot = "",
  [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Get-YamlLite([string]$path) {
  # Minimal YAML subset reader for our flat-ish configs (requires Python).
  python -c "import json,yaml,sys; print(json.dumps(yaml.safe_load(open(sys.argv[1],encoding='utf-8'))))" $path
}

$cfg = Get-YamlLite (Join-Path $Root $Config) | ConvertFrom-Json
if (-not $GgufDir) { $GgufDir = Join-Path $Root $cfg.gguf_dir }
if (-not $ShipRoot) { $ShipRoot = $cfg.ship_root }
$prefix = $cfg.model_name_prefix

if (-not (Test-Path $GgufDir)) { throw "GGUF dir missing: $GgufDir" }
New-Item -ItemType Directory -Force -Path $ShipRoot | Out-Null

$shipped = @()
Get-ChildItem $GgufDir -Filter "*.gguf" | ForEach-Object {
  $destDir = Join-Path $ShipRoot ("{0}-{1}" -f $prefix, $_.BaseName)
  New-Item -ItemType Directory -Force -Path $destDir | Out-Null
  $dest = Join-Path $destDir $_.Name
  if (Test-Path $dest) { Remove-Item -Force $dest }
  try {
    New-Item -ItemType HardLink -Path $dest -Target $_.FullName | Out-Null
  } catch {
    Copy-Item $_.FullName $dest -Force
  }
  # mmproj beside language model if present
  Get-ChildItem $_.Directory -Filter "*mmproj*" -ErrorAction SilentlyContinue | ForEach-Object {
    $mp = Join-Path $destDir $_.Name
    if (-not (Test-Path $mp)) {
      try { New-Item -ItemType HardLink -Path $mp -Target $_.FullName | Out-Null }
      catch { Copy-Item $_.FullName $mp -Force }
    }
  }
  $shipped += $destDir
  Write-Host "shipped $($_.Name) -> $destDir"
}

# Default load config snippet (document only — LM Studio concrete config path varies by build)
$defaults = @{
  draftMtp = [bool]$cfg.default_config.draftMtp
  kCache = [string]$cfg.default_config.kCache
  vCache = [string]$cfg.default_config.vCache
  parallel = [int]$cfg.runtime.parallel
  notes = @(
    "Do NOT enable --kv-mean-center with kCache=q8_0",
    "MTP: n-max=$($cfg.runtime.n_max) np=$($cfg.runtime.np) parallel=1"
  )
}
$defaultsPath = Join-Path $GgufDir "lms_default_load.json"
$defaults | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $defaultsPath
Write-Host "wrote $defaultsPath"

if (-not $SkipSmoke) {
  python (Join-Path $Root "scripts/smoke_prism.py") --config $Config
}

Write-Host "shipped $($shipped.Count) model dirs under $ShipRoot"
$shipped
