@echo off
cd /d "%~dp0"
"C:\Users\khura\AppData\Local\Programs\Python\Python311\python.exe" run_game.py
if errorlevel 1 pause
