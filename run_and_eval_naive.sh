#!/bin/bash
# ============================================================
# Glyph-Only (naive) benchmark & evaluation on UnseenWords
#
# 管线: Pass1 参考图 → VLM 布局 → Clean 背景 + 像素粘贴
# 无 latent 注入、频率分解、风格化
#
# 支持 base model:
#   naive_zimage : Z-Image    作为 Pass1 + Clean 背景
#   naive_qwen   : QwenImage  作为 Pass1 + Clean 背景
# ============================================================

# ==================== 可配置参数 ====================
RESULTS_VER="results_naive"
EVAL_VER="eval_results_naive"

BENCHMARK=UnseenWords
GPUS=8

# 基础路径
BASE_DIR="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher"
RESULTS_BASE="${BASE_DIR}/baselines/${RESULTS_VER}"
EVAL_BASE="${BASE_DIR}/${EVAL_VER}"

# 评估相关配置
METRICS="clip vlm vqa aesthetic"
BENCHMARK_DIR="eval/UnseenWords"
JSONL_FILES=(
    unseen_ez_en.jsonl unseen_ez_zh.jsonl unseen_ez_sci.jsonl unseen_en.jsonl
    unseen_zh.jsonl unseen_mid_sci.jsonl unseen_hardL1_sci.jsonl unseen_hardL2_sci.jsonl
)

# ==================== 实验配置 ====================
# 格式: "实验名:模型名:额外参数"
declare -a EXPERIMENTS=(
    "naive_zimage:glyph_only_zimage:"
    "naive_qwen:glyph_only_qwen:"
)

# ==================== 函数定义 ====================

run_experiment() {
    local exp_name=$1
    local model=$2
    local extra_args=$3
    local output_dir="${RESULTS_BASE}/${exp_name}/${BENCHMARK}"

    echo "============================================================"
    echo "Running: ${exp_name}  (model=${model})"
    echo "Output: ${output_dir}"
    echo "============================================================"

    cd "${BASE_DIR}/baselines" || exit 1

    python run_parallel_benchmark.py \
        --model ${model} \
        --benchmark $BENCHMARK \
        --gpus $GPUS \
        ${extra_args} \
        --output_dir ${output_dir} \
        --resume --skip-eval

    cd "$BASE_DIR" || exit 1
}

eval_experiment() {
    local exp_name=$1
    local results_dir="${RESULTS_BASE}/${exp_name}/${BENCHMARK}"
    local output_dir="${EVAL_BASE}/${exp_name}"

    echo "============================================================"
    echo "Evaluating: ${exp_name}"
    echo "  Results: $results_dir"
    echo "  Output : $output_dir"
    echo "============================================================"

    mkdir -p "$output_dir"

    for jsonl in "${JSONL_FILES[@]}"; do
        name="${jsonl%.jsonl}"
        python eval/scripts/eval_parallel.py \
            --results_dir "$results_dir" \
            --benchmark "${BENCHMARK_DIR}/${jsonl}" \
            --benchmark_type unseenwords \
            --output "${output_dir}/${name}.jsonl" \
            --metrics $METRICS \
            --gpus "$GPUS" \
            --resume \
            --verbose
    done
}

# ==================== 主流程 ====================

cd "$BASE_DIR" || exit 1
conda activate base

echo "============================================================"
echo "Glyph-Only (Naive) Benchmark & Evaluation"
echo "Results: ${RESULTS_VER}  |  Eval: ${EVAL_VER}"
echo "============================================================"

# 生成
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name model extra_args <<< "$exp"
    run_experiment "$exp_name" "$model" "$extra_args"
done

echo ""
echo "All experiments completed! Starting evaluation..."
conda activate t2v

# 评估
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name _ _ <<< "$exp"
    eval_experiment "$exp_name"
done

echo ""
echo "============================================================"
echo "All done!"
echo "Results: ${RESULTS_BASE}"
echo "Eval:    ${EVAL_BASE}"
echo "============================================================"
