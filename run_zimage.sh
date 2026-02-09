#!/bin/bash
# Z-Image 两阶段推理启动脚本
#
# 用法示例:
#   ./run_zimage.sh                                    # 正式模式（VLM 自主规划）
#   ./run_zimage.sh --debug-bypass                     # 调试旁路（手动 bbox）
#   ./run_zimage.sh --plan infer/test_plans/plan_multi_region.json  # JSON plan 测试
#   ./run_zimage.sh --tts --beam 8                     # 启用 TTS
#   ./run_zimage.sh --no-refiner                       # 不优化 prompt
#   ./run_zimage.sh --no-inject                        # 纯生图（不注入）

cd "$(dirname "$0")"

python run_zimage.py "$@"
