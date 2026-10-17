@echo off
cd /d "%~dp0"
echo =======================================================
echo Pushing LPG Smart Monitor to https://github.com/z1bfrr/-lpg-smart-monitor.git
echo =======================================================
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Code pushed successfully to GitHub!
) else (
    echo [ERROR] Push failed or was canceled. Please check your GitHub login.
)
echo =======================================================
pause
