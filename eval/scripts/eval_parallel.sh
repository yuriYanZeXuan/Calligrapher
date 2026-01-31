#!/bin/bash
# Parallel evaluation example with 8 GPUs

# Default paths
MINERU_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/MinerU_VLM"
VLM_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B"

# Example 1: Evaluate LongText-Bench with OCR + CLIP (8 GPUs)
echo "Running LongText-Bench evaluation..."
python eval/scripts/eval_parallel.py \
    --results_dir /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results/z_image/LongText-Bench \
    --benchmark /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/eval/LongText-Bench/text_prompts.jsonl \
    --benchmark_type longtext \
    --output eval_results/z_image_longtext2.jsonl \
    --metrics ocr clip vlm \
    --gpus 8 \
    --resume \
    --verbose

python eval/scripts/eval_parallel.py \
    --results_dir /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results/z_image/LongText-Bench \
    --benchmark /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/eval/LongText-Bench/text_prompt_zh.jsonl \
    --benchmark_type longtext \
    --output eval_results/z_image_longtext2.jsonl \
    --metrics ocr clip vlm \
    --gpus 8 \
    --resume \
    --verbose