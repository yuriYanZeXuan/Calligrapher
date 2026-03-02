#!/bin/bash
# ============================================================
# Frequency Decomposition & Attention Enhancement Ablation
# ============================================================

BASE_DIR="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher"
RESULTS_BASE="${BASE_DIR}/baselines/results_ablation"
BENCHMARK=UnseenWords
GPUS=8

# ==================== 实验配置 ====================
# 格式: "实验名:额外参数"
#
# 消融维度:
#   1. Harmonize:     w/ Klein  vs  w/o (no harmonize)
#   2. Freq Decomp:   w/ F.D.  vs  w/o F.D.
#   3. Attn Enhance:  w/ attn  vs  w/o attn
#
declare -a EXPERIMENTS=(
    # ========== Z-Image (--model ours) ==========
    # --- w/o Harmonize (Pass 2 直出) ---
    "ours:ablation_no_harmonize:--no-harmonize --no-refiner"

    # --- Full pipeline (Klein + F.D.) ---
    "ours:ablation_full_fd_klein:--freq-decompose --harmonizer-type klein --no-refiner"

    # --- w/o F.D. (Klein, 全频注入) ---
    "ours:ablation_no_fd_klein:--harmonizer-type klein --no-refiner"

    # --- w/ Attn Enhancement ---
    "ours:ablation_fd_attn_klein:--freq-decompose --harmonizer-type klein --no-refiner"

    # --- w/o Attn Enhancement ---
    "ours:ablation_fd_no_attn_klein:--freq-decompose --no-attn --harmonizer-type klein --no-refiner"

    # ========== QwenImage (--model ours_qwen) ==========
    # --- w/o Harmonize (Pass 2 直出) ---
    "ours_qwen:ablation_qwen_no_harmonize:--no-harmonize --no-refiner"

    # --- Full pipeline (Klein + F.D.) ---
    "ours_qwen:ablation_qwen_full_fd_klein:--freq-decompose --harmonizer-type klein --no-refiner"

    # --- w/o F.D. (Klein, 全频注入) ---
    "ours_qwen:ablation_qwen_no_fd_klein:--harmonizer-type klein --no-refiner"

    # --- w/ Attn Enhancement ---
    "ours_qwen:ablation_qwen_fd_attn_klein:--freq-decompose --harmonizer-type klein --no-refiner"

    # --- w/o Attn Enhancement ---
    "ours_qwen:ablation_qwen_fd_no_attn_klein:--freq-decompose --no-attn --harmonizer-type klein --no-refiner"
)

# ==================== 运行 ====================

cd "$BASE_DIR" || exit 1

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r model_name exp_name extra_args <<< "$exp"
    output_dir="${RESULTS_BASE}/${exp_name}/${BENCHMARK}"

    echo "============================================================"
    echo "Running: ${exp_name}  (model: ${model_name})"
    echo "  Args: ${extra_args}"
    echo "  Output: ${output_dir}"
    echo "============================================================"

    cd "${BASE_DIR}/baselines" || exit 1

    python run_parallel_benchmark.py \
        --model ${model_name} \
        --benchmark $BENCHMARK \
        --gpus $GPUS \
        ${extra_args} \
        --output_dir ${output_dir} \
        --resume --skip-eval

    cd "$BASE_DIR" || exit 1
done

echo ""
echo "============================================================"
echo "All ablation experiments completed!"
echo "Results: ${RESULTS_BASE}/ablation_*"
echo "============================================================"
