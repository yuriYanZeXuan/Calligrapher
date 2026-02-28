#!/bin/bash
# ============================================================
# 安装 Node.js (用户级, 无需 root) + 验证 MathJax 渲染
#
# 用法: bash scripts/setup_mathjax.sh
# ============================================================

set -e

NODE_VERSION="v20.18.3"
INSTALL_DIR="$HOME/.local"
ARCH=$(uname -m)

# x86_64 → x64, aarch64 → arm64
case "$ARCH" in
    x86_64)  NODE_ARCH="x64" ;;
    aarch64) NODE_ARCH="arm64" ;;
    *)       echo "不支持的架构: $ARCH"; exit 1 ;;
esac

TARBALL="node-${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz"
URL="https://nodejs.org/dist/${NODE_VERSION}/${TARBALL}"

echo "=== 安装 Node.js ${NODE_VERSION} (${NODE_ARCH}) ==="

# 确保 PATH 包含安装目录（无论是否已安装）
export PATH="$INSTALL_DIR/bin:$PATH"

# 持久化到所有 shell 启动文件（交互式 + 非交互式）
for rc in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.bash_profile"; do
    if [ -f "$rc" ] || [ "$rc" = "$HOME/.bashrc" ]; then
        if ! grep -q "$INSTALL_DIR/bin" "$rc" 2>/dev/null; then
            echo "export PATH=\"$INSTALL_DIR/bin:\$PATH\"" >> "$rc"
            echo "已写入 $rc"
        fi
    fi
done

# 检查是否已安装
if command -v node &>/dev/null; then
    echo "Node.js 已存在: $(node --version) at $(which node)"
    echo "跳过安装"
else
    mkdir -p "$INSTALL_DIR"
    TMP=$(mktemp -d)

    echo "下载 ${URL}..."
    wget -q --show-progress -O "$TMP/$TARBALL" "$URL"

    echo "解压到 ${INSTALL_DIR}..."
    tar -xJf "$TMP/$TARBALL" -C "$INSTALL_DIR" --strip-components=1

    rm -rf "$TMP"
    echo "Node.js 安装完成: $(node --version)"
fi

# 验证
echo ""
echo "=== 验证 MathJax 渲染 ==="
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MATHJAX_JS="${SCRIPT_DIR}/infer/render_mathjax.js"

if [ ! -f "$MATHJAX_JS" ]; then
    echo "错误: 找不到 ${MATHJAX_JS}"
    exit 1
fi

# 测试渲染
SVG=$(node "$MATHJAX_JS" '\nabla \cdot E = \rho' 2>&1)
if echo "$SVG" | grep -q '<svg'; then
    echo "MathJax 渲染成功!"
    echo "SVG 长度: ${#SVG} 字符"
else
    echo "MathJax 渲染失败:"
    echo "$SVG"
    exit 1
fi

echo ""
echo "=== 完成 ==="
echo "请重新运行: python infer/test_formula_render.py"
