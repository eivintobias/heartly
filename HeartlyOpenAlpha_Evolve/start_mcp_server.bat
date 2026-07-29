@echo off
REM ---------------------------------------------------------------
REM Launch HeartlyOpenAlpha_Evolve as a Gradio UI + MCP server.
REM   Web UI : http://127.0.0.1:7860
REM   MCP     : http://127.0.0.1:7860/gradio_api/mcp/
REM ---------------------------------------------------------------
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set GRADIO_MCP_SERVER=True
set GRADIO_SERVER_NAME=127.0.0.1
set GRADIO_SERVER_PORT=7860
set GRADIO_SHARE=false

python app.py
pause
