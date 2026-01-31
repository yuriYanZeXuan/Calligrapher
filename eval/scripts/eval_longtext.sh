#!/bin/bash
# LongText-Bench 评测脚本示例 - 支持本地权重加载

# 默认权重路径（可通过命令行覆盖）
MINERU_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/MinerU_VLM"
VLM_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B"

# 评测英文结果
python eval/scripts/evaluate.py \
    --mode generation \
    --benchmark eval/LongText-Bench/text_prompts.jsonl \
    --benchmark_type longtext \
    --generated baselines/results/z_image/LongText-Bench \
    --output eval_results/z_image_longtext_en.json \
    --metrics ocr clip \
    --mineru_path "$MINERU_PATH" \
    --vlm_path "$VLM_PATH"

# 评测中文结果
python eval/scripts/evaluate.py \
    --mode generation \
    --benchmark eval/LongText-Bench/text_prompts_zh.jsonl \
    --benchmark_type longtext \
    --generated baselines/results/z_image/LongText-Bench \
    --output eval_results/z_image_longtext_zh.json \
    --metrics ocr clip \
    --mineru_path "$MINERU_PATH" \
    --vlm_path "$VLM_PATH"

# 使用 VLM 评测（需要更多显存）
# python eval/scripts/evaluate.py \
#     --mode generation \
#     --benchmark eval/LongText-Bench/text_prompts.jsonl \
#     --benchmark_type longtext \
#     --generated baselines/results/z_image/LongText-Bench \
#     --output eval_results/z_image_longtext_vlm.json \
#     --metrics ocr clip vlm \
#     --mineru_path "$MINERU_PATH" \
#     --vlm_path "$VLM_PATH"
