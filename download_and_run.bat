@echo off
REM Complete Download and Run Script for Windows
REM This script downloads the repository from GitHub and runs everything automatically

echo ==========================================
echo Diabetes Readmission Prediction
echo Complete Download ^& Run Script
echo ==========================================
echo.

REM Repository URL
set REPO_URL=https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git
set REPO_NAME=diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-

REM Check if repository is already cloned
if exist "%REPO_NAME%" (
    echo [Repository folder already exists: %REPO_NAME%]
    echo    Using existing folder...
    cd "%REPO_NAME%"
    if errorlevel 1 (
        echo [ERROR] Cannot access repository folder!
        echo    Please check folder permissions or try running as administrator.
        pause
        exit /b 1
    )
) else (
    echo [Downloading repository from GitHub...]
    echo    URL: %REPO_URL%
    echo.
    
    REM Check if git is installed
    git --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Git is not installed!
        echo.
        echo Alternative: Download ZIP manually
        echo   1. Go to: %REPO_URL%
        echo   2. Click 'Code' -^> 'Download ZIP'
        echo   3. Extract the ZIP file
        echo   4. Rename folder to: %REPO_NAME%
        echo   5. Run this script again from that folder
        echo.
        echo Or install Git from: https://git-scm.com/download/win
        pause
        exit /b 1
    )
    
    REM Clone the repository
    git clone %REPO_URL% %REPO_NAME%
    
    if errorlevel 1 (
        echo [ERROR] Failed to clone repository!
        echo.
        echo Alternative: Download ZIP manually
        echo   1. Go to: %REPO_URL%
        echo   2. Click 'Code' -^> 'Download ZIP'
        echo   3. Extract the ZIP file
        echo   4. Rename folder to: %REPO_NAME%
        echo   5. Run this script again from that folder
        pause
        exit /b 1
    )
    
    if not exist "%REPO_NAME%" (
        echo [ERROR] Repository folder not found after cloning!
        echo    Expected: %CD%\%REPO_NAME%
        echo.
        echo Please try downloading ZIP manually instead.
        pause
        exit /b 1
    )
    
    echo [OK] Repository downloaded successfully!
    cd "%REPO_NAME%"
    if errorlevel 1 (
        echo [ERROR] Cannot access repository folder!
        echo    Path: %CD%\%REPO_NAME%
        echo    Please check folder permissions.
        pause
        exit /b 1
    )
)

echo.
echo [Working directory: %CD%]
echo.

REM Now run the setup and training script
echo ==========================================
echo [Starting Setup and Training]
echo ==========================================
echo.

REM Check if run_all.bat exists
if exist "run_all.bat" (
    echo [OK] Found run_all.bat script
    echo    Running automated setup...
    echo.
    call run_all.bat
) else (
    echo [WARNING] run_all.bat not found, running manual setup...
    echo.
    
    REM Manual setup steps
    REM Step 1: Check Python
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found! Please install Python 3.8+ first.
        pause
        exit /b 1
    )
    python --version
    echo [OK] Python found
    echo.
    
    REM Step 2: Create virtual environment
    echo [Setting up virtual environment...]
    if not exist ".venv" (
        python -m venv .venv
        echo [OK] Virtual environment created
    ) else (
        echo [OK] Virtual environment already exists
    )
    echo.
    
    REM Step 3: Activate and install
    echo [Activating virtual environment...]
    call .venv\Scripts\activate.bat
    if errorlevel 1 (
        echo [ERROR] Failed to activate virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment activated
    echo.
    
    echo [Installing required packages...]
    python -m pip install --upgrade pip --quiet
    python -m pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo [ERROR] Failed to install packages!
        pause
        exit /b 1
    )
    echo [OK] All packages installed
    echo.
    
    REM Step 4: Verify data
    if not exist "data\raw\diabetic_data.csv" (
        echo [ERROR] data\raw\diabetic_data.csv not found!
        pause
        exit /b 1
    )
    echo [OK] Data file found
    echo.
    
    REM Step 5: Run training
    echo [Starting training...]
    set OMP_NUM_THREADS=1
    python scripts\run_train.py
    if errorlevel 1 (
        echo [ERROR] Training failed!
        pause
        exit /b 1
    )
    echo.
    
    REM Step 6: Run evaluation
    echo [Running evaluation...]
    python scripts\run_eval.py
    if errorlevel 1 (
        echo [ERROR] Evaluation failed!
        pause
        exit /b 1
    )
    echo.
)

echo ==========================================
echo [OK] COMPLETE! Everything is ready!
echo ==========================================
echo.
echo [Project location: %CD%]
echo.
echo [Next steps:]
echo   1. View results in the terminal output above
echo   2. Launch dashboard:
echo      cd %CD%
echo      .venv\Scripts\activate
echo      streamlit run dashboard.py
echo.
echo ==========================================
pause

