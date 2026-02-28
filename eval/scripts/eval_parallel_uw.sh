#!/bin/bash
# ============================================================
# UnseenWords parallel evaluation — batch mode
#
# 一次加载所有 8 个 jsonl，每个 metric 只加载一次权重。
# 评测完成后按 source jsonl 自动拆分到 OUTPUT_DIR 下的
# 各子文件（unseen_en.jsonl, unseen_zh.jsonl, ...）。
#
# Usage:
#   bash eval/scripts/eval_parallel_uw.sh
# ============================================================

RESULTS_DIR="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results/fluxdev/UnseenWords"
OUTPUT_DIR="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/eval_results/fluxdev"
GPUS=8
METRICS="ocr clip vlm vqa aesthetic"

BENCHMARK_DIR="eval/UnseenWords"
MERGED_OUTPUT="${OUTPUT_DIR}/_merged_eval.jsonl"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "UnseenWords Evaluation (batch mode)"
echo "  Results dir : $RESULTS_DIR"
echo "  Benchmark   : $BENCHMARK_DIR (all jsonl files)"
echo "  Output dir  : $OUTPUT_DIR"
echo "  GPUs        : $GPUS"
echo "  Metrics     : $METRICS"
echo "============================================================"

python eval/scripts/eval_parallel.py \
    --results_dir "$RESULTS_DIR" \
    --benchmark "$BENCHMARK_DIR" \
    --benchmark_type unseenwords \
    --output "$MERGED_OUTPUT" \
    --metrics $METRICS \
    --gpus "$GPUS" \
    --resume \
    --verbose \
    --split_output_dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "All UnseenWords evaluations completed!"
echo "Per-jsonl summaries saved to: ${OUTPUT_DIR}/"
echo "============================================================"
