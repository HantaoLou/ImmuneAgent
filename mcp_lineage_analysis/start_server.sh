#!/bin/bash
# Influenza BCR Repertoire Analysis Server 启动脚本

# 设置工作目录
cd "$(dirname "$0")"

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 检查Python依赖
echo "检查Python依赖..."
python -c "import fastmcp" 2>/dev/null || {
    echo "安装依赖..."
    pip install -r requirements.txt
}

# 启动服务器
echo "启动 Influenza BCR Repertoire Analysis Server..."
python flu_bcr_repertoire_analysis_server.py

