#!/bin/bash
#
# 一键构建数据集脚本
# 用法: ./run_build.sh [web|ltb|all] [--debug]
#
# 支持 resume: 脚本会自动跳过已完成的任务
# 支持 --debug: 仅生成10条样本用于测试
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 解析参数
BUILD_TARGET="all"
DEBUG_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --debug)
            DEBUG_FLAG="--debug"
            ;;
        web|ltb|all)
            BUILD_TARGET="$arg"
            ;;
    esac
done

echo "========================================"
echo "数据集构建脚本"
echo "========================================"
echo "工作目录: $SCRIPT_DIR"
if [ -n "$DEBUG_FLAG" ]; then
    echo "模式: DEBUG (10条样本)"
fi
echo ""

case "$BUILD_TARGET" in
    web)
        echo "[任务] 仅构建 Web_rendered 数据集"
        python3 build_web_rendered.py $DEBUG_FLAG
        ;;
    ltb)
        echo "[任务] 仅构建 LTB_iid 数据集"
        python3 build_ltb_iid.py $DEBUG_FLAG
        ;;
    all)
        echo "[任务] 构建所有数据集"
        echo ""
        echo "========== 步骤 1/2: LTB_iid =========="
        python3 build_ltb_iid.py $DEBUG_FLAG &
        
        echo ""
        echo "========== 步骤 2/2: Web_rendered =========="
        python3 build_web_rendered.py $DEBUG_FLAG &
        ;;
    *)
        echo "用法: $0 [web|ltb|all] [--debug]"
        echo ""
        echo "  web     - 构建 Web_rendered 数据集"
        echo "  ltb     - 构建 LTB_iid 数据集"
        echo "  all     - 构建所有数据集 (默认)"
        echo "  --debug - Debug模式，仅生成10条样本"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "构建完成!"
echo "========================================"
