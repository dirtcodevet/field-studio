@echo off
rem Field Studio - field-based modelling (implicit fields, CPU only).
rem No arguments: opens the window with no console (pythonw); errors go to fieldstudio_last_run.log next to this file.
rem With arguments (e.g. --selftest, --monkey 120, --bust, --open X.fsproj, --ref-front F.jpg --ref-side S.jpg):
rem runs in this console so the output shows.
setlocal
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if "%~1"=="" (
  where pythonw >nul 2>nul && (
    start "" pythonw "%~dp0run_field_studio.py"
    goto :eof
  )
)
python "%~dp0run_field_studio.py" %*
if errorlevel 1 pause
endlocal
