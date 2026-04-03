@echo off
chcp 65001 >nul
echo ========================================
echo   SwarmGPT Extension - Windows Start
echo ========================================
echo.

REM Set API keys (replace with your actual keys)
set OPENAI_API_KEY=your-openai-api-key-here
set ANTHROPIC_API_KEY=your-anthropic-api-key-here

REM Change to script directory
cd /d "%~dp0"

echo Starting SwarmGPT...
echo Web interface will be available at: http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop
echo.

REM Activate conda environment and run
call D:\anaconda\Scripts\activate.bat swarmgpt
python swarm_gpt/launch.py

pause
