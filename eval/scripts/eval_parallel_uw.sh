#!/bin/bash
# ============================================================
# UnseenWords parallel evaluation — per-jsonl statistics
#
# Usage:
#   source eval/scripts/eval_parallel_uw.sh
# ============================================================

RESULTS_DIR="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results/fluxdev/UnseenWords"
OUTPUT_DIR="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/eval_results/fluxdev"
GPUS=8
METRICS="ocr clip vlm vqa aesthetic"

BENCHMARK_DIR="eval/UnseenWords"
JSONL_FILES=(
    unseen_ez_en.jsonl
    unseen_ez_zh.jsonl
    unseen_ez_sci.jsonl
    unseen_en.jsonl
    unseen_zh.jsonl
    unseen_mid_sci.jsonl
    unseen_hardL1_sci.jsonl
    unseen_hardL2_sci.jsonl
)

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "UnseenWords Evaluation"
echo "  Results dir : $RESULTS_DIR"
echo "  Output dir  : $OUTPUT_DIR"
echo "  GPUs        : $GPUS"
echo "  Metrics     : $METRICS"
echo "  Benchmarks  : ${#JSONL_FILES[@]} jsonl files"
echo "============================================================"

for jsonl in "${JSONL_FILES[@]}"; do
    name="${jsonl%.jsonl}"
    benchmark_path="${BENCHMARK_DIR}/${jsonl}"
    output_path="${OUTPUT_DIR}/${name}.jsonl"

    echo ""
    echo ">>> Evaluating: ${jsonl}"
    echo "    Output    : ${output_path}"

    python eval/scripts/eval_parallel.py \
        --results_dir "$RESULTS_DIR" \
        --benchmark "$benchmark_path" \
        --benchmark_type unseenwords \
        --output "$output_path" \
        --metrics $METRICS \
        --gpus "$GPUS" \
        --resume \
        --verbose
done

echo ""
echo "============================================================"
echo "All UnseenWords evaluations completed!"
echo "Per-jsonl summaries saved to: ${OUTPUT_DIR}/"
echo "============================================================"
