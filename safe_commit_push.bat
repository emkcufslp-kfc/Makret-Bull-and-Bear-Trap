@echo off
setlocal
cd /d "D:\Backtest\Makret-Bull-and-Bear-Trap"

echo === Stopping fsmonitor daemon and removing stale git locks ===
git fsmonitor--daemon stop 2>nul
if exist ".git\index.lock" del /f ".git\index.lock"
if exist ".git\HEAD.lock" del /f ".git\HEAD.lock"
if exist ".git\refs\heads\main.lock" del /f ".git\refs\heads\main.lock"
echo Locks cleared.
echo.

echo === Staging your local changes (with line-ending normalization) ===
git add --renormalize .
git add -A

echo === Committing your local changes ===
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
    echo Nothing new to commit locally - will still sync with GitHub Actions below.
)
echo.

echo === Fetching latest from GitHub (this pulls in the automated daily-refresh bot commits) ===
git fetch origin
if errorlevel 1 (
    echo ERROR: fetch failed - check your internet connection / GitHub access
    pause
    exit /b 1
)
echo.

echo === Merging bot updates in, your local changes win on any conflicting file ===
git merge origin/main -X ours -m "Merge automated data refresh (local changes take precedence on conflicts)"
if errorlevel 1 (
    echo ERROR: merge failed - resolve manually with "git status" / "git mergetool", then re-run this script.
    pause
    exit /b 1
)
echo.

echo === Pushing merged result to GitHub (no force needed - nothing gets overwritten) ===
git push origin main
if errorlevel 1 (
    echo ERROR: push failed - check output above
    pause
    exit /b 1
)

echo.
echo SUCCESS - your changes and the bot's automated refresh are both on GitHub now.
pause
