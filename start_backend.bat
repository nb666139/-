@echo off
echo === GridSynergy Backend ===
cd /d "%~dp0submission"
echo Installing Python dependencies...
pip install -r requirements.txt
echo Starting API server on http://localhost:8888
python web/server_lite.py
pause
