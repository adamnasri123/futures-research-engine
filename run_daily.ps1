# Launches the autonomous trader each weekday morning.
# DRY-RUN by default (no real orders). Change the last line to add --live ONLY
# after several clean dry-run days (see STRATEGY.md).
#
# Register with Windows Task Scheduler so it runs hands-free — see SETUP below.

Set-Location -Path $PSScriptRoot
# LIVE: places real bracketed orders on the TopStep eval account.
& "$PSScriptRoot\venv\Scripts\python.exe" "$PSScriptRoot\autotrader.py" --live
# To dry-run instead (no orders), remove the --live flag above.
