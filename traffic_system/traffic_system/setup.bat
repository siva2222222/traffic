@echo off
REM ===================================================================
REM  AI Traffic Management System - one-time setup (Windows, Python 3.11)
REM  Run this ONCE:   setup.bat
REM  Then every time: run.bat
REM ===================================================================
setlocal

echo.
echo [1/4] Checking Python 3.11...
py -3.11 --version
if errorlevel 1 (
  echo.
  echo ERROR: Python 3.11 not found.
  echo Install it from https://www.python.org/downloads/release/python-3119/
  echo Tick "Add python.exe to PATH" during install, then run setup.bat again.
  pause
  exit /b 1
)

echo.
echo [2/4] Creating virtual environment "venv"...
if not exist venv (
  py -3.11 -m venv venv
) else (
  echo     venv already exists - reusing it
)

echo.
echo [3/4] Installing packages (this downloads ~1.5 GB, takes 10-20 min)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Package install failed. Most common cause is "lap".
  echo Trying the prebuilt replacement instead...
  pip install lapx
)

echo.
echo [4/4] Verifying the install...
python check_setup.py

echo.
echo ===================================================================
echo  Setup finished. Start the app with:   run.bat
echo ===================================================================
pause
