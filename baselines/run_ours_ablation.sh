#!/bin/bash
# ============================================================
# Ours method benchmark & ablation experiments on UnseenWords
#
# Ablation variants:
#   ours_full        : all components enabled
#   ours_no_inject   : w/o glyph injector  (--no-inject)
#   ours_no_harmonize: w/o FluxKlein refine (--no-harmonize), pass2 as final
#   ours_no_refiner  : w/o prompt refiner   (--no-refiner)
# ============================================================


BENCHMARK=UnseenWords
GPUS=8

# ---- Full model (all components enabled) ----
python run_parallel_benchmark.py \
    --model ours \
    --benchmark $BENCHMARK \
    --gpus $GPUS \
    --output_dir results/ours_full/$BENCHMARK\
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
