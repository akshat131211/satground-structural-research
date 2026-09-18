$ErrorActionPreference = 'Stop'
$trainingReviewRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $trainingReviewRoot
& (Join-Path $trainingReviewRoot '.venv\python.exe') (Join-Path $PSScriptRoot 'open_training_review.py') --open-browser
