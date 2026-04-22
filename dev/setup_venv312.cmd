@echo off
setlocal
cd /d %~dp0\..
powershell -NoProfile -ExecutionPolicy Bypass -File dev\setup_venv312.ps1 %*
