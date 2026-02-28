#!/usr/bin/env bash
# ------------------------------------------------------------------
# Incremental evaluation script with hardcoded paths.
#
# Step 1: Construct detail.jsonl for each results dir (if missing).
# Step 2: Run parallel evaluation with CLIP, VQA, vlm_quality metrics.
#
# Supports resume — safe to re-run after interruption.
# ------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_BASE="baselines/results"
EVAL_DIR="eval"
OUTPUT_BASE="eval_results"
GPUS="${GPUS:-8}"

# ---- Benchmark configurations ----
declare -A BENCHMARKS
BENCHMARKS["LongText-Bench"]="${EVAL_DIR}/LongText-Bench:longtext"
BENCHMARKS["OneIG-Bench"]="${EVAL_DIR}/OneIG-Bench:oneig"
BENCHMARKS["UnseenWords"]="${EVAL_DIR}/UnseenWords:unseenwords"

# ---- Models to evaluate ----
MODELS=(
    "ours"
    "ours_qwen"
    "glyph_only_zimage"
    "glyph_only_qwen"
    "textflux"
    "textcrafter_flux"
    "anytext"
    "qwenedit"
    "fluxfill"
    "fluxdev"
    "fluxklein"
    "glm_image"
    "z_image"
    "qwenimage"
    "nanobanana"
    "fluxtext"
)

# ---- Our models that have VLM-generated clean_prompt ----
OUR_MODELS="ours ours_qwen glyph_only_zimage glyph_only_qwen glyph_only_klein"

is_our_model() {
    local model="$1"
    for m in $OUR_MODELS; do
        [[ "$m" == "$model" ]] && return 0
    done
    return 1
}

# ---- Main loop ----
for model in "${MODELS[@]}"; do
    for bench_name in "${!BENCHMARKS[@]}"; do
        IFS=":" read -r bench_path bench_type <<< "${BENCHMARKS[$bench_name]}"
        results_dir="${RESULTS_BASE}/${model}/${bench_name}"

        # Skip if results dir does not exist
        if [[ ! -d "$results_dir" ]]; then
            echo "[SKIP] $results_dir does not exist"
            continue
        fi

        echo ""
        echo "============================================================"
        echo " Model: $model | Benchmark: $bench_name"
        echo "============================================================"

        # Step 1: Construct detail.jsonl if not present
        detail_path="${results_dir}/detail.jsonl"
        if [[ ! -f "$detail_path" ]]; then
            echo "[Step 1] Constructing detail.jsonl ..."
            vlm_flag=""
            if is_our_model "$model"; then
                vlm_flag="--use_vlm"
            fi
            python scripts/construct_detail.py \
                --results_dir "$results_dir" \
                --benchmark "$bench_path" \
                --benchmark_type "$bench_type" \
                --resume \
                $vlm_flag
        else
            echo "[Step 1] detail.jsonl already exists, skipping construction"
        fi

        # Step 2: Parallel evaluation (CLIP, VQA, vlm_quality)
        output_jsonl="${OUTPUT_BASE}/${model}/${bench_name}/eval_detail.jsonl"
        mkdir -p "$(dirname "$output_jsonl")"

        echo "[Step 2] Evaluating CLIP, VQA, vlm_quality, hpsv3 (resume mode) ..."
        python eval/scripts/eval_parallel.py \
            --results_dir "$results_dir" \
            --benchmark "$bench_path" \
            --benchmark_type "$bench_type" \
            --output "$output_jsonl" \
            --metrics clip vqa vlm_quality hpsv3 \
            --gpus "$GPUS" \
            --resume \
            --verbose

        echo "[DONE] $model / $bench_name"
    done
done

echo ""
echo "========================================"
echo " All evaluations complete."
echo "========================================"
