@echo off
echo Starting Airline Robustness App...
python run_demo.py
if errorlevel 1 (
    echo.
    echo Error occurred. Please make sure Python is installed and requirements are installed.
    echo Try running: pip install -r requirements.txt
    pause
)
pause
