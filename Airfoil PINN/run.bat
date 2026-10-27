@echo off
cd /d "%~dp0"
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set "AIRFOIL_PINN_PYTHON_ENV=D:\Python\BaseEnv"
set "AIRFOIL_PYTHON=%AIRFOIL_PINN_PYTHON_ENV%\Scripts\python.exe"

if not exist "%AIRFOIL_PYTHON%" (
    echo [ERROR] Python environment not found: %AIRFOIL_PINN_PYTHON_ENV%
    echo The project does not create environments or install packages automatically.
    pause
    exit /b 1
)

"%AIRFOIL_PYTHON%" Airfoil_PINN.py
pause
