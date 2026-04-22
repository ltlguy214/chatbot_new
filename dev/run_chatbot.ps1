$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot

$Py = Join-Path $RepoRoot '.venv312\Scripts\python.exe'
if (!(Test-Path $Py)) {
  throw 'Missing .venv312. Run ./dev/setup_venv312.ps1 first.'
}

& $Py -m streamlit run chatbot/app_chatbot.py
