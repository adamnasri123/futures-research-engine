# Starts the local trading dashboard and opens it in the default browser.
Set-Location -Path (Split-Path $PSScriptRoot -Parent)
Start-Process "http://127.0.0.1:8765"
& ".\venv\Scripts\python.exe" -m dashboard.app
