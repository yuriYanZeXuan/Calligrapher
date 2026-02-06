#!/bin/bash
# 安装脚本 - 配置所有依赖

echo "========================================"
echo "LTB 数据集构建工具 - 依赖安装"
echo "========================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "✗ Python3 未安装"
    exit 1
fi
echo "✓ Python3: $(python3 --version)"

# 安装 Python 依赖
echo ""
echo "[1/3] 安装 Python 依赖..."
pip3 install -q openai python-dotenv markdown playwright pyyaml
echo "✓ Python 依赖安装完成"

# 安装 Playwright 浏览器
echo ""
echo "[2/3] 安装 Playwright 浏览器..."
playwright install chromium
echo "✓ Playwright 浏览器安装完成"

# 下载 MathJax（如果不存在）
echo ""
echo "[3/3] 检查 MathJax..."
if [ -d "assets/mathjax" ]; then
    echo "✓ MathJax 已存在"
else
    echo "下载 MathJax..."
    mkdir -p assets
    cd assets
    curl -L -o mathjax.tar.gz https://github.com/mathjax/MathJax/archive/refs/tags/3.2.2.tar.gz
    tar -xzf mathjax.tar.gz
    mv MathJax-3.2.2 mathjax
    rm mathjax.tar.gz
    cd ..
    echo "✓ MathJax 下载完成"
fi

echo ""
echo "========================================"
echo "✓ 所有依赖安装完成！"
echo "========================================"
echo ""
echo "下一步："
echo "  1. 配置 .env 文件（API密钥）"
echo "  2. 运行 ./run_build.sh --debug 测试"
echo ""
