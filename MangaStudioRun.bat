@echo off
REM =================================================================
REM ==                                                             ==
REM ==              MANGA TRANSLATION STUDIO - LAUNCHER              ==
REM ==                                                             ==
REM =================================================================

REM Set the working directory to this script's location.
cd /d "%~dp0"

:MENU
cls
echo.
echo  =========================================
echo       MANGA TRANSLATION STUDIO LAUNCHER
echo  =========================================
echo.
echo  Please choose a launch mode:
echo.
echo    [1] Normal Mode (No Console Window)
echo.
echo    [2] Debug Mode  (With Console Window)
echo.
echo    [Q] Quit
echo.
echo  =========================================
echo.

CHOICE /C 12Q /M "Enter your choice: "

IF ERRORLEVEL 3 GOTO END
IF ERRORLEVEL 2 GOTO DEBUG_MODE
IF ERRORLEVEL 1 GOTO NORMAL_MODE

:NORMAL_MODE
echo [INFO] Launching in Normal (Silent) Mode...
start "" ".\venv\Scripts\pythonw.exe" MangaStudio.py
GOTO END

:DEBUG_MODE
echo [INFO] Launching in Debug (Console) Mode...
echo -----------------------------------------------------------------
.\venv\Scripts\python.exe MangaStudio.py
echo -----------------------------------------------------------------
echo [DEBUG] Script has finished or was stopped.
pause
GOTO END

:END
exit