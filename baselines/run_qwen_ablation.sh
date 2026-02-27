#!/bin/bash
# ============================================================
# QwenImage-base method benchmark & ablation on UnseenWords
#
# Ablation variants (mirror run_ours_ablation.sh):
#   qwen_full         : all components enabled (klein harmonizer)
#   qwen_no_inject    : w/o glyph injector  (--no-inject)
#   qwen_no_harmonize : w/o Pass 3 refine   (--no-harmonize)
#   qwen_no_refiner   : w/o prompt refiner  (--no-refiner)
#   qwen_freq_decomp  : w/ freq decomposition (--freq-decompose)
# ============================================================

BENCHMARK=UnseenWords
GPUS=8

# ---- Full model (klein harmonizer) ----
python run_parallel_benchmark.py \
    --model ours_qwen \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --output_dir results/qwen_full/$BENCHMARK \
    --resume --skip-eval

# ---- Ablation: w/o glyph injector ----
python run_parallel_benchmark.py \
    --model ours_qwen \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --no-inject \
    --output_dir results/qwen_no_inject/$BENCHMARK \
    --resume --skip-eval

# ---- Ablation: w/o Pass 3 refine ----
python run_parallel_benchmark.py \
    --model ours_qwen \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --no-harmonize \
    --output_dir results/qwen_no_harmonize/$BENCHMARK \
    --resume --skip-eval

# ---- Ablation: w/o prompt refiner ----
python run_parallel_benchmark.py \
    --model ours_qwen \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --no-refiner \
    --output_dir results/qwen_no_refiner/$BENCHMARK \
    --resume --skip-eval

# ---- Ablation: w/ freq decomposition ----
python run_parallel_benchmark.py \
    --model ours_qwen \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --freq-decompose \
    --output_dir results/qwen_freq_decomp/$BENCHMARK \
    --resume --skip-eval
