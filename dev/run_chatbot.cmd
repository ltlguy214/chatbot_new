@echo off
setlocal
cd /d %~dp0\..
powershell -NoProfile -ExecutionPolicy Bypass -File dev\run_chatbot.ps1 %*
