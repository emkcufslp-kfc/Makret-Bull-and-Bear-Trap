@echo off
cd /d "D:\Backtest\Makret-Bull-and-Bear-Trap"

echo === Stopping fsmonitor daemon and removing stale git locks ===
git fsmonitor--daemon stop 2>nul
if exist ".git\index.lock" del /f ".git\index.lock"
if exist ".git\HEAD.lock" del /f ".git\HEAD.lock"
if exist ".git\refs\heads\main.lock" del /f ".git\refs\heads\main.lock"
echo Locks cleared.
echo.

echo === Staging changes (with line-ending normalization) ===
git add --renormalize .
git add -A

echo === Committing ===
set MSG=%~1
if "%MSG%"=="" set MSG=Update dashboards and strategies
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "%MSG%"
    if errorlevel 1 (
        echo ERROR: commit FAILED with staged changes - NOT pushing. Fix the error above first.
        pause
        exit /b 1
    )
) else (
    echo Nothing staged to commit - pushing existing commits only.
)
echo.

echo === Pushing to GitHub (overwriting bot data commits) ===
git fetch origin
git push --force-with-lease origin main
if errorlevel 1 (
    echo ERROR: push failed - check output above
    pause
    exit /b 1
)

echo.
echo SUCCESS - committed and pushed to GitHub!
pause
