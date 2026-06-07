@echo off
chcp 65001 >nul
echo ==========================================
echo     外呼 Agent 评测平台 - 启动中...
echo ==========================================

cd /d "%~dp0"

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

echo ✅ Python 已安装

REM 检查依赖
echo 📦 检查依赖...
pip install -r requirements.txt -q 2>nul

REM 检查 .env
if not exist .env (
    echo ⚠️ 未找到 .env 文件，正在创建...
    (
        echo OPENAI_API_KEY=your_api_key_here
        echo BASE_URL=https://api.deepseek.com
        echo LLM_MODEL=deepseek-chat
        echo JUDGE_LLM_MODEL=deepseek-chat
    ) > .env
    echo 📝 请编辑 .env 文件填入你的 API Key
)

echo.
echo 🚀 启动 Dashboard...
echo ==========================================
echo.

python -m outbound_eval serve --host 0.0.0.0 --port 8000

pause
