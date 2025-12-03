@echo off
REM Complete Setup and Run Script for Windows
REM This script does everything: setup, install, train, evaluate

echo ==========================================
echo Diabetes Readmission Prediction - Setup ^& Run
echo ==========================================
echo.

REM Get the directory where this script is located
cd /d "%~dp0"

echo [Project directory: %CD%]
echo.

REM Step 1: Check Python
echo [Checking Python installation...]
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)
python --version
echo [OK] Python found
echo.

REM Step 2: Create virtual environment if it doesn't exist
echo [Setting up virtual environment...]
if not exist ".venv" (
    echo    Creating new virtual environment...
    python -m venv .venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Step 3: Activate virtual environment
echo [Activating virtual environment...]
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Step 4: Upgrade pip
echo [Upgrading pip...]
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded
echo.

REM Step 5: Install requirements
echo [Installing required packages...]
echo    (This may take a few minutes...)
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install packages!
    pause
    exit /b 1
)
echo [OK] All packages installed
echo.

REM Step 6: Verify data file exists
echo [Checking data files...]
if not exist "data\raw\diabetic_data.csv" (
    echo [ERROR] data\raw\diabetic_data.csv not found!
    echo    Please make sure the data file is in the correct location.
    pause
    exit /b 1
)
echo [OK] Data file found
echo.

REM Step 7: Run training
echo ==========================================
echo [Starting Training Process]
echo ==========================================
echo.
echo This will:
echo   - Clean and preprocess the data
echo   - Train Logistic Regression model
echo   - Train XGBoost model
echo   - Save all models and processed data
echo.
echo [Training in progress... (This takes 2-5 minutes)]
echo.

REM Set OMP_NUM_THREADS to avoid potential issues
set OMP_NUM_THREADS=1

python scripts\run_train.py
if errorlevel 1 (
    echo.
    echo [ERROR] Training failed! Please check the error messages above.
    pause
    exit /b 1
)
echo.
echo [OK] Training completed successfully!
echo.

REM Step 8: Run evaluation
echo ==========================================
echo [Running Model Evaluation]
echo ==========================================
echo.
echo [Evaluating models... (This takes 1-2 minutes)]
echo.

python scripts\run_eval.py
if errorlevel 1 (
    echo.
    echo [ERROR] Evaluation failed! Please check the error messages above.
    pause
    exit /b 1
)
echo.
echo [OK] Evaluation completed successfully!
echo.

REM Step 9: Summary
echo ==========================================
echo [OK] ALL STEPS COMPLETED SUCCESSFULLY!
echo ==========================================
echo.
echo [Generated files:]
echo   - data\processed\train_processed.csv
echo   - data\processed\test_processed.csv
echo   - models\logreg_selected.joblib
echo   - models\xgb_selected.joblib
echo   - models\thresholds.json
echo.
echo [Next steps:]
echo   1. View results in the terminal output above
echo   2. Launch dashboard: streamlit run dashboard.py
echo   3. Open notebooks: jupyter lab notebooks\03_implementation_details.ipynb
echo.
echo [To run dashboard, use:]
echo   .venv\Scripts\activate
echo   streamlit run dashboard.py
echo.
echo ==========================================
pause

