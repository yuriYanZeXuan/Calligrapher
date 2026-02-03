#!/bin/bash
#
# 一键构建数据集脚本
# 用法: ./run_build.sh [web|ltb|all]
#
# 支持 resume: 脚本会自动跳过已完成的任务
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "数据集构建脚本"
echo "========================================"
echo "工作目录: $SCRIPT_DIR"
echo ""

# 默认构建全部
BUILD_TARGET="${1:-all}"

case "$BUILD_TARGET" in
    web)
        echo "[任务] 仅构建 Web_rendered 数据集"
        python3 build_web_rendered.py
        ;;
    ltb)
        echo "[任务] 仅构建 LTB_iid 数据集"
        python3 build_ltb_iid.py
        ;;
    all)
        echo "[任务] 构建所有数据集"
        echo ""
        echo "========== 步骤 1/2: LTB_iid =========="
        python3 build_ltb_iid.py
        
        echo ""
        echo "========== 步骤 2/2: Web_rendered =========="
        python3 build_web_rendered.py
        ;;
    *)
        echo "用法: $0 [web|ltb|all]"
        echo ""
        echo "  web  - 构建 Web_rendered 数据集"
        echo "  ltb  - 构建 LTB_iid 数据集"
        echo "  all  - 构建所有数据集 (默认)"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "构建完成!"
echo "========================================"
