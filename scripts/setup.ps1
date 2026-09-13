$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    conda create --prefix '.venv' python=3.10 pip --override-channels --channel conda-forge --yes
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
& $pythonPath -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
if (Test-Path -LiteralPath 'requirements-resolved.txt') {
    & $pythonPath -m pip install -r requirements-resolved.txt
} else {
    & $pythonPath -m pip install -r requirements-model.txt -e .
}
if ($LASTEXITCODE -ne 0) { throw 'Research dependency installation failed.' }
& $pythonPath scripts/freeze_dependencies.py
& $pythonPath -m satground bootstrap
if ($LASTEXITCODE -ne 0) { throw 'Asset bootstrap failed.' }
& $pythonPath -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
