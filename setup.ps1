$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.python'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache'
uv venv --python 3.11 .venv
if ($LASTEXITCODE -ne 0) { throw 'Python environment setup failed.' }
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
Write-Host 'Setup complete. Add GOOGLE_API_KEY to .env, start Docker Desktop, then run .\start.ps1.'
