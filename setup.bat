@echo off
echo ==========================================
echo Initializing Python Environment (Windows)
echo ==========================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10 or higher.
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment and install dependencies
echo Activating .venv and installing dependencies...
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)

echo.
echo ==========================================
echo Setup successful! 
echo To activate the environment manually, run:
echo .venv\Scripts\activate
echo ==========================================
pause
