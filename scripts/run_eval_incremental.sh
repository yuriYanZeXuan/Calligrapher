#!/usr/bin/env bash
# ------------------------------------------------------------------
# Incremental evaluation script.
#
# 直接提供 results 目录的绝对路径，脚本会：
#   1. 在该目录下构造 detail.jsonl（如果缺失）
#   2. 在该目录下运行增量评测，输出 eval_detail.jsonl
#
# 从目录名自动推断 benchmark 类型，从路径关键词推断是否调用 VLM。
# 支持 resume — 中断后可安全重跑。
# ------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

GPUS="${GPUS:-8}"
EVAL_DIR="eval"

# ---- Benchmark 数据路径 (从目录末尾名自动匹配) ----
declare -A BENCH_PATH
BENCH_PATH["LongText-Bench"]="${EVAL_DIR}/LongText-Bench"
BENCH_PATH["OneIG-Bench"]="${EVAL_DIR}/OneIG-Bench"
BENCH_PATH["UnseenWords"]="${EVAL_DIR}/UnseenWords"
BENCH_PATH["CVTG-2K"]="${EVAL_DIR}/CVTG-2K"

declare -A BENCH_TYPE
BENCH_TYPE["LongText-Bench"]="longtext"
BENCH_TYPE["OneIG-Bench"]="oneig"
BENCH_TYPE["UnseenWords"]="unseenwords"
BENCH_TYPE["CVTG-2K"]="cvtg"

# 从路径关键词判断是否为 "我们的模型"（需要 VLM 生成 clean_prompt）
is_our_model_path() {
    local p="$1"
    [[ "$p" == *"/ours"* || "$p" == *"/glyph_only"* ]] && return 0
    return 1
}

# ============================================================
#  在此列出所有待评测的 results 目录（绝对路径）
#  脚本会从最后一级目录名推断 benchmark 类型。
# ============================================================
RESULT_DIRS=(
    /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results7/ours_freq_decomp_klein/UnseenWords
    /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results7/ours_full_klein/UnseenWords
    /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results7/ours_no_harmonize/UnseenWords
    /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results7/ours_no_inject/UnseenWords
    # 按需添加更多目录，例如:
    # /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results7/ours_freq_decomp_klein/LongText-Bench
    # /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/results7/ours_freq_decomp_klein/OneIG-Bench
)

# ---- Main loop ----
for results_dir in "${RESULT_DIRS[@]}"; do
    if [[ ! -d "$results_dir" ]]; then
        echo "[SKIP] $results_dir does not exist"
        continue
    fi

    # 从目录路径最后一级推断 benchmark 名
    bench_name="$(basename "$results_dir")"
    if [[ -z "${BENCH_PATH[$bench_name]+x}" ]]; then
        echo "[ERROR] Unknown benchmark '$bench_name' (from $results_dir), skipping"
        continue
    fi
    bench_path="${BENCH_PATH[$bench_name]}"
    bench_type="${BENCH_TYPE[$bench_name]}"

    echo ""
    echo "============================================================"
    echo " Dir:       $results_dir"
    echo " Benchmark: $bench_name ($bench_type)"
    echo "============================================================"

    # Step 1: 在 results_dir 下构造 detail.jsonl
    detail_path="${results_dir}/detail.jsonl"
    if [[ ! -f "$detail_path" ]]; then
        echo "[Step 1] Constructing detail.jsonl ..."
        vlm_flag=""
        if is_our_model_path "$results_dir"; then
            vlm_flag="--use_vlm"
        fi
        python scripts/construct_detail.py \
            --results_dir "$results_dir" \
            --benchmark "$bench_path" \
            --benchmark_type "$bench_type" \
            --resume \
            $vlm_flag
    else
        echo "[Step 1] detail.jsonl already exists, skipping"
    fi

    # Step 2: 在 results_dir 下运行评测，输出也保存在同目录
    output_jsonl="${results_dir}/eval_detail.jsonl"

    echo "[Step 2] Evaluating clip, vqa, vlm_quality, hpsv3 (resume) ..."
    python eval/scripts/eval_parallel.py \
        --results_dir "$results_dir" \
        --benchmark "$bench_path" \
        --benchmark_type "$bench_type" \
        --output "$output_jsonl" \
        --metrics clip vqa vlm_quality hpsv3 \
        --gpus "$GPUS" \
        --resume \
        --verbose

    echo "[DONE] $results_dir"
done

echo ""
echo "========================================"
echo " All evaluations complete."
echo "========================================"
