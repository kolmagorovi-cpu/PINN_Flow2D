@echo off
cd /d "%~dp0"
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
rem Активируем общее виртуальное окружение PINN_Flow2D (если есть), иначе локальное venv\
if exist ..\.venv\Scripts\activate.bat (
    call ..\.venv\Scripts\activate.bat
) else (
    call venv\Scripts\activate.bat
)
python pipe_bend_main.py
pause
