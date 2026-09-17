$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& (Join-Path $projectRoot '.venv\python.exe') (Join-Path $PSScriptRoot 'open_review_batch.py') --open-browser
