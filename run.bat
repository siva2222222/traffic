@echo off
REM Start the AI Traffic Management System. Run setup.bat first (once).
setlocal
if not exist venv (
  echo venv not found - run setup.bat first.
  pause
  exit /b 1
)
call venv\Scripts\activate.bat
echo.
echo Starting server... open http://127.0.0.1:5000/admin/image in your browser.
echo Press Ctrl+C here to stop.
echo.
python app.py
pause
