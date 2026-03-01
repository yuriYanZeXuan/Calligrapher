#!/usr/bin/env bash
# ------------------------------------------------------------------
# Incremental evaluation script — Multi-Dir Batch Mode.
#
# 按 benchmark 类型分组，将同类型的所有 results 目录合并到一次
# eval_parallel.py 调用中，每个指标只加载一次模型权重。
#
# 评测完成后，结果自动拆分回各目录的 eval_detail.jsonl（格式不变）。
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

# ---- 按 benchmark 类型分组，批量评测 ----
echo ""
echo "========================================"
echo " Batch evaluation (multi-dir)"
echo "========================================"

# 按 bench_name 分组
declare -A GROUPED_DIRS  # bench_name -> space-separated dirs

for results_dir in "${RESULT_DIRS[@]}"; do
    if [[ ! -d "$results_dir" ]]; then
        continue
    fi
    bench_name="$(basename "$results_dir")"
    if [[ -z "${BENCH_PATH[$bench_name]+x}" ]]; then
        continue
    fi
    GROUPED_DIRS["$bench_name"]="${GROUPED_DIRS[$bench_name]:-} $results_dir"
done

for bench_name in "${!GROUPED_DIRS[@]}"; do
    bench_path="${BENCH_PATH[$bench_name]}"
    bench_type="${BENCH_TYPE[$bench_name]}"
    # 将空格分隔的路径转为数组
    read -ra dirs <<< "${GROUPED_DIRS[$bench_name]}"

    echo ""
    echo "============================================================"
    echo " Benchmark: $bench_name ($bench_type)"
    echo " Dirs (${#dirs[@]}):"
    for d in "${dirs[@]}"; do
        echo "   - $d"
    done
    echo "============================================================"

    if [[ ${#dirs[@]} -eq 1 ]]; then
        # 单目录回退到 --results_dir 模式
        python eval/scripts/eval_parallel.py \
            --results_dir "${dirs[0]}" \
            --benchmark "$bench_path" \
            --benchmark_type "$bench_type" \
            --output "${dirs[0]}/eval_detail.jsonl" \
            --metrics clip vqa vlm_quality hpsv3 \
            --gpus "$GPUS" \
            --resume \
            --verbose
    else
        # 多目录合并评测（每个指标只加载一次权重）
        python eval/scripts/eval_parallel.py \
            --results_dirs "${dirs[@]}" \
            --benchmark "$bench_path" \
            --benchmark_type "$bench_type" \
            --metrics clip vqa vlm_quality hpsv3 \
            --gpus "$GPUS" \
            --resume \
            --verbose
    fi
done

echo ""
echo "========================================"
echo " All evaluations complete."
echo "========================================"
