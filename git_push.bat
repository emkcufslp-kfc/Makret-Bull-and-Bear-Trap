@echo off
cd /d "D:\Backtest\Makret-Bull-and-Bear-Trap"

echo === Removing stale git lock files ===
if exist ".git\index.lock" (
    del /f ".git\index.lock"
    echo Removed index.lock
)
if exist ".git\HEAD.lock" (
    del /f ".git\HEAD.lock"
    echo Removed HEAD.lock
)
if exist ".git\refs\heads\main.lock" (
    del /f ".git\refs\heads\main.lock"
    echo Removed main.lock
)
echo.

echo === Staging all changes ===
git add -A
if errorlevel 1 (
    echo ERROR: git add failed
    pause
    exit /b 1
)

echo === Committing ===
git diff --cached --quiet
if not errorlevel 1 (
    echo Nothing to commit, skipping.
) else (
    git commit -m "Daily refresh: signal caches, exports, data updates"
    if errorlevel 1 (
        echo ERROR: git commit failed
        pause
        exit /b 1
    )
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
