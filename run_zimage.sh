#!/bin/bash
# Z-Image 三阶段推理启动脚本
#
# 三阶段推理架构:
#   Pass 1: 用完整 prompt 生成参考图 → VLM 自主规划排版
#   Pass 2: 从相同噪声用 clean prompt + 字形注入生成文字图
#   Pass 3: FluxKlein img2img + 二值 mask → 将文字风格化为白色粉笔字效果
#
# 用法示例:
#   ./run_zimage.sh                                    # 正式模式（VLM 自主规划）
#   ./run_zimage.sh --plan infer/test_plans/plan_multi_region.json  # JSON plan 测试
#
# 功能开关:
#   ./run_zimage.sh --no-refiner                       # 禁用 prompt 优化
#   ./run_zimage.sh --no-inject                        # 禁用字形注入（纯生图）
#   ./run_zimage.sh --no-harmonize                     # 跳过 Pass 3 风格化
#
# 注入参数调节:
#   ./run_zimage.sh --mask-strength 0.8                # 调整 mask 注入强度 (默认 1.0)
#   ./run_zimage.sh --strength-schedule cosine         # 注入强度衰减策略: constant/linear/cosine
#   ./run_zimage.sh --attn-enhance 1.5                 # 注意力增强倍率 (默认 2.0)
#   ./run_zimage.sh --attn-suppress 0.2                # 反向注意力抑制倍率 (默认 0.1)
#
# FluxKlein 参数:
#   ./run_zimage.sh --klein-steps 20                   # FluxKlein 推理步数 (默认 10)
#   ./run_zimage.sh --klein-guidance 3.0               # FluxKlein guidance scale (默认 4.0)
#
# 基础参数:
#   ./run_zimage.sh --prompt "A poster with ..." --text "E=mc^2" "Another text"
#   ./run_zimage.sh --steps 30 --seed 123 --height 768 --width 1024
#   ./run_zimage.sh --gpus 0,1                         # 指定 GPU

cd "$(dirname "$0")"

python run_zimage.py "$@"
