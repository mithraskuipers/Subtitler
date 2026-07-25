@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ============================================
echo  Subtitler - Startup
echo ============================================
echo.

REM Some dependencies (pydantic-core, av) do not yet ship prebuilt wheels for
REM Python 3.13/3.14, which makes pip try to compile them from source and
REM fail on a machine without Rust/FFmpeg dev headers set up. To avoid that,
REM this app always runs on Python 3.12, regardless of what else is on PATH.

set "PYTHON_LAUNCHER="
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 -c "" >nul 2>nul
    if !errorlevel!==0 set "PYTHON_LAUNCHER=py -3.12"
)

if not defined PYTHON_LAUNCHER (
    where python3.12 >nul 2>nul
    if %errorlevel%==0 set "PYTHON_LAUNCHER=python3.12"
)

if not defined PYTHON_LAUNCHER (
    echo Python 3.12 was not found on this system.
    echo This app requires Python 3.12 specifically ^(newer versions such as
    echo 3.13/3.14 currently fail to install some dependencies^).
    echo Install it from https://www.python.org/downloads/release/python-3120/
    echo and make sure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    set "VENV_PY_VERSION="
    for /f "tokens=2 delims= " %%v in ('"%VENV_PYTHON%" --version 2^>^&1') do set "VENV_PY_VERSION=%%v"
    echo !VENV_PY_VERSION! | findstr /b "3.12." >nul
    if not !errorlevel!==0 (
        echo Existing virtual environment uses Python !VENV_PY_VERSION! - recreating it with Python 3.12 ...
        rmdir /s /q "%VENV_DIR%"
    )
)

if not exist "%VENV_PYTHON%" (
    echo No virtual environment found - creating one in .venv ...
    %PYTHON_LAUNCHER% -m venv "%VENV_DIR%"
    if not exist "%VENV_PYTHON%" (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Using existing virtual environment: %VENV_DIR%
)

REM Every run of this script goes through the venv's own Python from here on,
REM never the system Python, so dependencies stay isolated to this app.

set "REQUIREMENTS_STAMP=%VENV_DIR%\.requirements.stamp"
set "REQUIREMENTS_CHANGED=1"

if exist "%REQUIREMENTS_STAMP%" (
    fc /b "requirements.txt" "%REQUIREMENTS_STAMP%" >nul 2>nul
    if !errorlevel!==0 set "REQUIREMENTS_CHANGED=0"
)

if "%REQUIREMENTS_CHANGED%"=="1" (
    echo.
    echo Installing/updating dependencies inside the virtual environment ...
    "%VENV_PYTHON%" -m pip install --upgrade pip >nul
    "%VENV_PYTHON%" -m pip install -r requirements.txt
    if not !errorlevel!==0 (
        echo.
        echo Failed to install dependencies. Check your internet connection and try again.
        pause
        exit /b 1
    )
    copy /y "requirements.txt" "%REQUIREMENTS_STAMP%" >nul
) else (
    echo Dependencies already up to date - skipping reinstall.
)

echo.
echo Speech-recognition models are stored locally in the "models" folder next
echo to this script. Nothing is downloaded automatically - use the Models
echo panel in the app to download a model before running a batch.
echo.
echo Starting the application ...
echo A browser window will open automatically once the server is ready.
echo Close this window to stop the application.
echo.

"%VENV_PYTHON%" -m backend.main

pause
