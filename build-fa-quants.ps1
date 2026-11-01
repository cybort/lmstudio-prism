#Requires -Version 5.1
<#
.SYNOPSIS
  Build Prism llama.cpp CUDA with extra asymmetric FlashAttention KV kernels + kv-mean-center.

.DESCRIPTION
  Clones PrismML-Eng/llama.cpp at the pin (default prism-b10777-e3de715) into vendor/llama.cpp
  if missing, configures GGML_CUDA_FA_QUANTS for:
    q8_0-q4_0, f16-q8_0, f16-q4_0, bf16-q8_0, bf16-q4_0
  plus the stock symmetric pairs, then builds Release into vendor/cuda-fa.

  After build: .\install.ps1 -Force  (config local_path -> vendor/cuda-fa)
#>
param(
  [string]$Tag = "prism-b10777-e3de715",
  [string]$CudaArch = "89",
  [string]$FaQuants = "q4_0-q4_0;q8_0-q8_0;q8_0-q4_0;f16-f16;f16-q8_0;f16-q4_0;bf16-bf16;bf16-q8_0;bf16-q4_0",
  [switch]$ConfigureOnly
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Vendor = Join-Path $Root "vendor"
$Src = Join-Path $Vendor "llama.cpp"
$Build = Join-Path $Vendor "build-cuda-fa"
$Stage = Join-Path $Vendor "cuda-fa"

New-Item -ItemType Directory -Force -Path $Vendor | Out-Null

if (-not (Test-Path (Join-Path $Src ".git"))) {
  Write-Host "[build-fa] cloning PrismML-Eng/llama.cpp @$Tag"
  git clone --depth 1 --branch $Tag https://github.com/PrismML-Eng/llama.cpp.git $Src
}

Write-Host "[build-fa] FA_QUANTS=$FaQuants ARCH=$CudaArch"
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
New-Item -ItemType Directory -Force -Path $Build | Out-Null

cmake -S $Src -B $Build -G "Visual Studio 18 2026" -A x64 `
  -DGGML_CUDA=ON `
  -DGGML_CUDA_FA=ON `
  "-DGGML_CUDA_FA_QUANTS=$FaQuants" `
  "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch" `
  -DLLAMA_BUILD_SERVER=ON `
  -DLLAMA_CURL=OFF `
  -DGGML_RPC=ON
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

if ($ConfigureOnly) {
  Write-Host "[build-fa] configure only; done"
  exit 0
}

cmake --build $Build --config Release --parallel --target llama-server llama-kv-mean-center
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

$bin = Join-Path $Build "bin\Release"
if (-not (Test-Path (Join-Path $bin "ggml-cuda.dll"))) {
  throw "ggml-cuda.dll missing under $bin"
}

if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
Copy-Item (Join-Path $bin "*") $Stage -Force

Write-Host "[build-fa] staged -> $Stage"
Write-Host "[build-fa] next: set config.json local_path to that folder, then .\install.ps1 -Force"
Write-Host "[build-fa] optional: .\make-kv-bias.ps1"
Write-Host "STAGE_DIR=$Stage"
