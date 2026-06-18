@echo off
cd /d "%~dp0"
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
call venv\Scripts\activate.bat
python Airfoil_PINN.py
pause
