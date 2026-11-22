# Build the Windows onedir package for CARTLab.
# Run from the repository root on Windows 11 with Python 3.11 and uv installed.
$ErrorActionPreference = 'Stop'

uv sync --locked --group dev

uv run --locked --group dev pyinstaller `
  --noconfirm --clean --onedir --windowed `
  --name CARTLab `
  --add-data "frontend;frontend" `
  --add-data "datasets;datasets" `
  main.py

Write-Host "Output: dist\CARTLab\CARTLab.exe"
