@echo off
echo Starting Wildfire Prediction Application...

:: Check Python installation
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    pause
    exit /b
)

:: Check Poetry installation
poetry --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Poetry is not installed or not in PATH. Please install Poetry and try again.
    pause
    exit /b
)

:: Install dependencies using Poetry
echo Installing dependencies with Poetry...
poetry install

:: Start FastAPI backend
echo Starting FastAPI backend...
start cmd /k "poetry run uvicorn src.main:app --port 8060 --reload"

:: Start Streamlit frontend
echo Starting Streamlit frontend...
start cmd /k "poetry run streamlit run frontend/frontend.py --server.port 8501 --server.headless true"

echo Application started! Open the following URLs:
echo - FastAPI Docs: http://localhost:8060/docs
echo - Streamlit Frontend: http://localhost:8501
pause