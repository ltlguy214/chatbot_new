param(
  [switch]$UseLock,
  [switch]$Recreate
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot

$VenvDir = Join-Path $RepoRoot '.venv312'
$Py = Join-Path $VenvDir 'Scripts\python.exe'

if ($Recreate -and (Test-Path $VenvDir)) {
  Write-Host "Removing existing venv at $VenvDir" -ForegroundColor Yellow
  Remove-Item -Recurse -Force $VenvDir
}

if (!(Test-Path $Py)) {
  Write-Host "Creating venv at $VenvDir" -ForegroundColor Cyan
  # Requires Python 3.12 installed via Python Launcher (py). If not available, fallback to `python`.
  $pyLauncher = (Get-Command py -ErrorAction SilentlyContinue)
  if ($pyLauncher) {
    py -3.12 -m venv $VenvDir
  } else {
    python -m venv $VenvDir
  }
}

# Ensure the venv is Python 3.12.x (this repo was developed on 3.12.10).
$ver = & $Py -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"
if (-not $ver.StartsWith('3.12.')) {
  throw "Expected Python 3.12.x in .venv312, got $ver. Install Python 3.12 and re-run setup."
}

Write-Host 'Upgrading pip/setuptools/wheel…' -ForegroundColor Cyan
& $Py -m pip install --upgrade pip setuptools wheel

$lockFile = Join-Path $RepoRoot 'requirements-venv312.lock.txt'
if ($UseLock -and (Test-Path $lockFile)) {
  Write-Host "Installing from lockfile: $lockFile" -ForegroundColor Cyan
  & $Py -m pip install -r $lockFile
} else {
  Write-Host 'Installing from chatbot/requirements.txt' -ForegroundColor Cyan
  & $Py -m pip install -r (Join-Path $RepoRoot 'chatbot\requirements.txt')
}

Write-Host 'Done. Next: Copy chatbot/.env.example -> chatbot/.env' -ForegroundColor Green
