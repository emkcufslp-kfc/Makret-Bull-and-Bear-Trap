@echo off
cd /d "D:\Backtest\Makret-Bull-and-Bear-Trap"

echo === Removing stale git lock if present ===
if exist ".git\index.lock" (
    del /f ".git\index.lock"
    echo Lock removed.
) else (
    echo No lock file found.
)
echo.

echo === Pushing to GitHub ===
git push --force origin main
if errorlevel 1 (
    echo ERROR: push failed - check output above
    pause
    exit /b 1
)

echo.
echo SUCCESS - pushed to GitHub!
pause
