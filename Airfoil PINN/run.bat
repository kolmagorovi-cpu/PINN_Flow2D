@echo off
setlocal
cd /d "%~dp0"
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
if defined AIRFOIL_PINN_PYTHON_ENV (
    set "AIRFOIL_PYTHON=%AIRFOIL_PINN_PYTHON_ENV%\Scripts\python.exe"
    goto run
)
if exist "D:\Python\BaseEnv\Scripts\python.exe" (
    set "AIRFOIL_PYTHON=D:\Python\BaseEnv\Scripts\python.exe"
    goto run
)
if exist "%~dp0venv\Scripts\python.exe" (
    set "AIRFOIL_PYTHON=%~dp0venv\Scripts\python.exe"
    goto run
)
rem Bootstrap creates local venv using an installed Python.
where python >nul 2>nul
if not errorlevel 1 (
    python Airfoil_PINN.py %*
    goto done
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 Airfoil_PINN.py %*
    goto done
)
echo [ERROR] Python is not installed or is not available in PATH.
pause
exit /b 1
:run
if not exist "%AIRFOIL_PYTHON%" (
    echo [ERROR] Python environment not found: %AIRFOIL_PINN_PYTHON_ENV%
    pause
    exit /b 1
)
"%AIRFOIL_PYTHON%" Airfoil_PINN.py %*
:done
set "AIRFOIL_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %AIRFOIL_EXIT_CODE%
