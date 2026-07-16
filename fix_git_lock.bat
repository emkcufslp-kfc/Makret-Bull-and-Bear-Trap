@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo  Git Lock / Stray Staging Fixer
echo  Repo: %cd%
echo ============================================
echo.

REM --- Step 1: warn about processes that might hold the lock open ---
echo [1/4] Checking for git processes...
tasklist /fi "imagename eq git.exe" 2>nul | find /i "git.exe" >nul
if %errorlevel%==0 (
    echo   Found running git.exe process^(es^):
    tasklist /fi "imagename eq git.exe"
    echo.
    set /p KILLIT="  Kill these git.exe processes now? (y/n): "
    if /i "!KILLIT!"=="y" (
        taskkill /f /im git.exe >nul 2>&1
        echo   Killed.
    ) else (
        echo   Skipped. If the lock removal below fails, close the process
        echo   holding it (git GUI, editor, terminal) and re-run this script.
    )
) else (
    echo   No git.exe process found running.
)
echo.

REM --- Step 2: remove stale lock files ---
echo [2/4] Removing stale lock files in .git\...
if exist ".git\index.lock" (
    del /f /q ".git\index.lock" 2>nul
    if exist ".git\index.lock" (
        echo   FAILED to delete .git\index.lock - it is still in use.
        echo   Close whatever program has it open ^(git GUI, VS Code, etc.^)
        echo   and run this script again.
        echo.
        pause
        exit /b 1
    ) else (
        echo   Removed .git\index.lock
    )
) else (
    echo   No .git\index.lock present.
)
if exist ".git\HEAD.lock" (
    del /f /q ".git\HEAD.lock" 2>nul
    echo   Removed .git\HEAD.lock
)
for %%R in (main) do (
    if exist ".git\refs\heads\%%R.lock" (
        del /f /q ".git\refs\heads\%%R.lock" 2>nul
        echo   Removed .git\refs\heads\%%R.lock
    )
)
echo.

REM --- Step 3: unstage everything (safe - does not touch working files) ---
echo [3/4] Resetting the index (unstaging everything, working files untouched)...
git reset
if errorlevel 1 (
    echo   git reset failed - see error above.
    pause
    exit /b 1
)
echo   Done. Nothing is staged now.
echo.

REM --- Step 4: show status for manual review ---
echo [4/4] Current status - REVIEW BEFORE COMMITTING:
echo ============================================
git status
echo ============================================
echo.
echo Nothing has been committed or pushed by this script.
echo Review the list above. If it looks correct (only real changes,
echo no unexpected deletions of source files), stage and commit
echo with your normal workflow, e.g.:
echo.
echo     safe_commit_push.bat
echo.
echo or manually:
echo.
echo     git add data\r3_signal_cache.json
echo     git add exports\crash_predictor_study\weekly_feature_outcomes.csv
echo     git commit -m "chore: daily refresh"
echo     git push origin main
echo.
pause
