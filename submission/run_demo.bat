@echo off
chcp 65001 >nul
echo ============================================
echo   GridSynergy - Demo Starter
echo ============================================
echo.

echo [1/2] Running experiments...
python experiments\run_experiments.py --mode all
echo.

echo [2/2] Starting Web Server (http://localhost:8080)...
start "GridSynergy API" python web\api.py

echo.
echo Opening browser...
start "" demo\index.html

echo ============================================
echo   System Started!
echo   API Docs: http://localhost:8080/docs
echo ============================================
pause
