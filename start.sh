#!/bin/bash
# 外呼 Agent 评测平台 - 一键启动脚本

echo "=========================================="
echo "    外呼 Agent 评测平台 - 启动中..."
echo "=========================================="

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python3，请先安装 Python 3.10+"
    exit 1
fi

echo "✅ Python 版本: $(python3 --version)"

# 检查依赖
echo "📦 检查依赖..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "正在安装依赖..."
    pip3 install -r requirements.txt -q
fi

# 检查 .env 配置
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件，正在创建..."
    cp .env.example .env 2>/dev/null || cat > .env << 'EOF'
OPENAI_API_KEY=your_api_key_here
BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
JUDGE_LLM_MODEL=deepseek-chat
EOF
    echo "📝 请编辑 .env 文件填入你的 API Key"
fi

echo ""
echo "🚀 启动 Dashboard..."
echo "=========================================="
echo ""

# 启动服务
python3 -m outbound_eval serve --host 0.0.0.0 --port 8000
