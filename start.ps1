$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Run the setup steps in README.md first.'
}
& '.\.venv\Scripts\python.exe' app.py
