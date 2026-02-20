#!/bin/bash
# ============================================================
# Ours method benchmark & ablation experiments on UnseenWords
#
# Ablation variants:
#   ours_full        : all components enabled (klein harmonizer)
#   ours_qwenedit    : use QwenEdit as Pass 3 harmonizer
#   ours_no_inject   : w/o glyph injector  (--no-inject)
#   ours_no_harmonize: w/o Pass 3 refine    (--no-harmonize), pass2 as final
#   ours_no_refiner  : w/o prompt refiner   (--no-refiner)
#   ours_freq_decomp : w/ freq decomposition (--freq-decompose)
# ============================================================


BENCHMARK=UnseenWords
GPUS=8

# ---- Full model (klein harmonizer) ----
python run_parallel_benchmark.py \
    --model ours \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --output_dir results/ours_full/$BENCHMARK \
    --debug

# ---- Full model (qwenedit harmonizer) ----
python run_parallel_benchmark.py \
    --model ours \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --harmonizer-type qwenedit \
    --output_dir results/ours_qwenedit/$BENCHMARK \
    --debug

# ---- Ablation: w/o glyph injector ----
python run_parallel_benchmark.py \
    --model ours \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --no-inject \
    --output_dir results/ours_no_inject/$BENCHMARK \
    --debug

# ---- Ablation: w/o FluxKlein refine (pass2 as final) ----
python run_parallel_benchmark.py \
    --model ours \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --no-harmonize \
    --output_dir results/ours_no_harmonize/$BENCHMARK \
    --debug

# ---- Ablation: w/o prompt refiner ----
python run_parallel_benchmark.py \
    --model ours \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --no-refiner \
    --output_dir results/ours_no_refiner/$BENCHMARK \
    --debug

# ---- Ablation: w/ freq decomposition ----
python run_parallel_benchmark.py \
    --model ours \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --freq-decompose \
    --harmonizer-type qwenedit \
    --output_dir results/ours_freq_decomp/$BENCHMARK \
    --debug
