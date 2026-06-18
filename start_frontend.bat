@echo off
echo === GridSynergy Frontend ===
cd /d "%~dp0frontend"
echo Installing dependencies...
call npm install
echo Starting dev server on http://localhost:5173
call npm run dev
pause
