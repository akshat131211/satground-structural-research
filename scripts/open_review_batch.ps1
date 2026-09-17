$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$reviewView = Read-Host 'Choose a review view from 1 to 18 (Enter for 1)'
if ([string]::IsNullOrWhiteSpace($reviewView)) { $reviewView = '1' }
if ($reviewView -notmatch '^([1-9]|1[0-8])$') { throw 'Enter a whole number from 1 to 18.' }
& (Join-Path $projectRoot '.venv\python.exe') (Join-Path $PSScriptRoot 'open_review_batch.py') --view $reviewView --open-browser
