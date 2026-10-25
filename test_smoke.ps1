#Requires -Version 5.1
<#
.SYNOPSIS
  Smoke-test the installed Prism LM Studio backend (or local_path) with PQ2_0.
#>
param(
  [string]$ConfigPath = "",
  [string]$BackendDir = ""
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
if (-not $ConfigPath) { $ConfigPath = Join-Path $Root "config.json" }
if (-not (Test-Path $ConfigPath)) {
  Copy-Item (Join-Path $Root "config.example.json") $ConfigPath
}
$cfg = Get-Content $ConfigPath -Raw | ConvertFrom-Json

if (-not $BackendDir) {
  $installTarget = if ($cfg.install_target) { $cfg.install_target } else { "overlay" }
  if ($installTarget -eq "overlay") {
    $BackendDir = Join-Path $env:USERPROFILE ".lmstudio\extensions\backends\$($cfg.donor_backend)"
  }
  else {
    $BackendDir = Join-Path $env:USERPROFILE ".lmstudio\extensions\backends\$($cfg.backend_name)-$($cfg.backend_version)"
  }
}
if (-not (Test-Path (Join-Path $BackendDir "llama-server.exe"))) {
  Write-Host "[smoke] backend missing; falling back to local_path"
  $BackendDir = $cfg.local_path
}

$exe = Join-Path $BackendDir "llama-server.exe"
$model = $cfg.model_for_test
$port = if ($cfg.smoke_port) { [int]$cfg.smoke_port } else { 18082 }
$ngl = if ($cfg.smoke_ngl) { [int]$cfg.smoke_ngl } else { 8 }

if (-not (Test-Path $exe)) { throw "llama-server.exe not found: $exe" }
if (-not (Test-Path $model)) { throw "model not found: $model" }

$log = Join-Path $Root "cache\smoke.err.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
if (Test-Path $log) { Remove-Item $log -Force }

Write-Host "[smoke] exe=$exe"
Write-Host "[smoke] model=$model"
Write-Host "[smoke] port=$port ngl=$ngl"

$args = @(
  "-m", $model,
  "--host", "127.0.0.1",
  "--port", "$port",
  "-c", "2048",
  "-ngl", "$ngl",
  "--fit", "off",
  "--reasoning", "off",
  "--reasoning-budget", "0"
)

$p = Start-Process -FilePath $exe -ArgumentList $args -RedirectStandardError $log -PassThru -WindowStyle Hidden
Write-Host "[smoke] pid=$($p.Id)"

$ok = $false
try {
  for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 5
    if ($p.HasExited) { throw "server exited early code=$($p.ExitCode); see $log" }
    try {
      $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3 -UseBasicParsing
      if ($r.StatusCode -eq 200) { $ok = $true; Write-Host "[smoke] healthy at ${i}x5s"; break }
    } catch {}
  }
  if (-not $ok) { throw "health check timed out; see $log" }

  $body = '{"model":"x","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":8,"temperature":0}'
  $chat = Invoke-WebRequest -Uri "http://127.0.0.1:$port/v1/chat/completions" `
    -Method POST -Body $body -ContentType "application/json" -TimeoutSec 300 -UseBasicParsing
  Write-Host "[smoke] response:"
  Write-Host $chat.Content

  $json = $chat.Content | ConvertFrom-Json
  $text = $json.choices[0].message.content
  if (-not $text) { throw "empty completion" }
  Write-Host "[smoke] PASS content=$text"
  exit 0
}
finally {
  if ($p -and -not $p.HasExited) {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
  }
}
