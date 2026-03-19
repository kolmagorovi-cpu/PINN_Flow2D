@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python Airfoil_PINN.py
pause
