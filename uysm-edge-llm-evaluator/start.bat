@echo off
echo Baslatiyor...

:: Backend
start "Backend" cmd /k "cd /d %~dp0 && uvicorn web_backend.main:app --reload --port 8000"

:: Frontend
start "Frontend" cmd /k "cd /d %~dp0web_frontend && npm run dev"

echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
timeout /t 3 >nul
start http://localhost:5173
