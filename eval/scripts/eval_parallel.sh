#!/bin/bash
# Parallel evaluation example with 8 GPUs

# Default paths
MINERU_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/MinerU_VLM"
VLM_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B"

# Example 1: Evaluate LongText-Bench with OCR + CLIP (8 GPUs)
echo "Running LongText-Bench evaluation..."
python eval/scripts/eval_parallel.py \
    --results_dir /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results/z_image/LongText-Bench \
    --benchmark /Users/yanzexuan/code/Calligrapher/eval/LongText-Bench/text_prompts.jsonl \
    --benchmark_type longtext \
    --output eval_results/z_image_longtext.jsonl \
    --metrics ocr clip \
    --gpus 8 \
    --resume \
    --verbose

# Example 2: Resume evaluation (only evaluate remaining samples)
# echo "Resuming evaluation..."
# python eval/scripts/eval_parallel.py \
#     --results_dir /path/to/results \
#     --benchmark /path/to/benchmark.jsonl \
#     --benchmark_type longtext \
#     --output eval_results/output.jsonl \
#     --metrics ocr clip \
#     --gpus 8 \
#     --resume

# Example 3: Include VLM evaluation (slower, more GPU memory)
# echo "Running with VLM evaluation..."
# python eval/scripts/eval_parallel.py \
#     --results_dir /path/to/results \
#     --benchmark /path/to/benchmark.jsonl \
#     --benchmark_type longtext \
#     --output eval_results/with_vlm.jsonl \
#     --metrics ocr clip vlm \
#     --gpus 8 \
#     --mineru_path "$MINERU_PATH" \
#     --vlm_path "$VLM_PATH" \
#     --resume

# Example 4: Single GPU mode (for testing)
# python eval/scripts/eval_parallel.py \
#     --results_dir /path/to/results \
#     --benchmark /path/to/benchmark.jsonl \
#     --benchmark_type longtext \
#     --output eval_results/test.jsonl \
#     --metrics ocr \
#     --gpus 1
