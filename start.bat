@echo off
REM SwarmGPT Windows Startup Script

echo ========================================
echo   SwarmGPT Extension - Quick Start
echo ========================================
echo.

REM Activate conda environment
call conda activate swarmgpt
if %errorlevel% neq 0 (
    echo Error: Failed to activate conda environment 'swarmgpt'
    echo Please make sure the environment exists: conda create -n swarmgpt python=3.11
    pause
    exit /b 1
)

REM Check for API key
if "%OPENAI_API_KEY%"=="" (
    echo Warning: OPENAI_API_KEY is not set.
    echo Please set your API key:
    echo   set OPENAI_API_KEY=sk-...
    echo.
    echo Or create a key.bat file with:
    echo   set OPENAI_API_KEY=sk-...
    echo.
)

REM Change to project directory
cd /d "%~dp0"

REM Run tests first
echo Running tests...
python -m pytest tests/ -q --ignore=tests/test_providers --ignore=tests/unit/test_backend.py 2>nul
if %errorlevel% neq 0 (
    echo Warning: Some tests failed, but continuing...
)

echo.
echo Starting SwarmGPT...
echo Web interface will be available at: http://127.0.0.1:7860
echo.

REM Launch SwarmGPT
python swarm_gpt/launch.py

pause
