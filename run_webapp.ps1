param(
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not in PATH. Install Python 3.11+ and retry."
    exit 1
}

$venvDir = Join-Path $root "venv"
$venvPython = Join-Path $venvDir "Scripts/python.exe"
$venvPip = Join-Path $venvDir "Scripts/pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at $venvDir"
    python -m venv $venvDir
}

"" | Out-Null
$depsMarker = Join-Path $venvDir ".deps_installed"
if (-not (Test-Path $depsMarker)) {
    Write-Host "Installing dependencies (first run)..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r "$root/requirements.txt" flask pillow
    New-Item -ItemType File -Path $depsMarker -Force | Out-Null
} else {
    Write-Host "Dependencies already installed; skipping reinstall."
}

# Ensure audio model exists
$audioCkpt = Join-Path $root "checkpoints/audio_kaggle_best.pt"
if (-not (Test-Path $audioCkpt)) {
    Write-Warning "Audio checkpoint missing: $audioCkpt"
}
# HuggingFace image model will download on first run into checkpoints/pretrained_hf

Write-Host "Starting web app on port $Port ..."
$env:FLASK_RUN_PORT = $Port
& $venvPython "$root/web/app.py"
