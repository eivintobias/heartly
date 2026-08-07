@echo off
cd /d "%~dp0"
echo =================================================
echo  Heartly Qwen-Code v3 server (FastAPI / llama.cpp-ready)
echo  Swagger UI: http://127.0.0.1:8000/docs
echo  CPU-only | first /chat loads the ~3 GB weights once (~3s), then fast
echo  Close this window to STOP the server.
echo =================================================
timeout /t 6 >nul
start "" "http://127.0.0.1:8000/docs"
"C:\Users\eivin\AppData\Local\Programs\Python\Python312\python.exe" server.py --model heartly-qwen-code-v3 --host 127.0.0.1 --port 8000
