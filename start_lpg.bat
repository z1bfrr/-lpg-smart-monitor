@echo off
cd /d "%~dp0"
echo ========================================================
echo  Starting LPG Smart Monitor Dashboard on http://localhost:5173
echo ========================================================
start "" "http://localhost:5173"
npm run dev
pause
