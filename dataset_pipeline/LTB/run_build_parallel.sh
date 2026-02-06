#!/bin/bash
# 并行构建脚本 - 同时运行 LTB_iid 和 Web_rendered
# 用法: ./run_build_parallel.sh [--debug] [--ltb-samples N] [--web-samples N]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DEBUG_MODE=0
LTB_SAMPLES=""
WEB_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) DEBUG_MODE=1; shift ;;
        --ltb-samples) LTB_SAMPLES="$2"; shift 2 ;;
        --web-samples) WEB_SAMPLES="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "并行数据集构建"
echo "========================================"

LTB_ARGS=""
WEB_ARGS=""

if [ $DEBUG_MODE -eq 1 ]; then
    LTB_ARGS="--debug"
    WEB_ARGS="--debug"
    echo "模式: DEBUG (10条/语种)"
elif [ -n "$LTB_SAMPLES" ] || [ -n "$WEB_SAMPLES" ]; then
    [ -n "$LTB_SAMPLES" ] && LTB_ARGS="--samples $LTB_SAMPLES" && echo "LTB: $LTB_SAMPLES 条/语种"
    [ -n "$WEB_SAMPLES" ] && WEB_ARGS="--samples $WEB_SAMPLES" && echo "Web: $WEB_SAMPLES 条/语种"
fi

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "启动并行构建..."
python3 build_ltb_iid.py $LTB_ARGS > "logs/ltb_$TIMESTAMP.log" 2>&1 &
LTB_PID=$!
python3 build_web_rendered.py $WEB_ARGS > "logs/web_$TIMESTAMP.log" 2>&1 &
WEB_PID=$!

echo "LTB_iid PID: $LTB_PID"
echo "Web_rendered PID: $WEB_PID"
echo ""
echo "监控: tail -f logs/ltb_$TIMESTAMP.log"
echo "      tail -f logs/web_$TIMESTAMP.log"
echo ""

wait $LTB_PID
LTB_EXIT=$?
wait $WEB_PID
WEB_EXIT=$?

echo ""
[ $LTB_EXIT -eq 0 ] && echo "[✓] LTB_iid 完成" || echo "[✗] LTB_iid 失败"
[ $WEB_EXIT -eq 0 ] && echo "[✓] Web_rendered 完成" || echo "[✗] Web_rendered 失败"
echo "========================================"

exit $(( LTB_EXIT + WEB_EXIT ))
