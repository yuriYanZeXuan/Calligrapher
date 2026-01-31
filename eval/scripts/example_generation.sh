#!/bin/bash
# Example: Generation Evaluation Scripts
# 
# These are example commands for running generation evaluation.
# Modify the paths according to your setup.

set -e

# Configuration
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"

echo "==================================="
echo "Generation Evaluation Examples"
echo "==================================="
echo ""

# Example 1: OneIG-Bench with Text_Rendering category
echo "Example 1: OneIG-Bench Text Rendering Evaluation"
echo "------------------------------------------------"
python "$PROJECT_ROOT/eval/scripts/evaluate.py" \
    --mode generation \
    --benchmark "$PROJECT_ROOT/eval/OneIG-Bench/OneIG-Bench.json" \
    --benchmark_type oneig \
    --filter_categories Text_Rendering \
    --generated "$PROJECT_ROOT/outputs/oneig_results" \
    --output "$PROJECT_ROOT/eval_results/oneig_text_rendering.json" \
    --metrics ocr clip \
    --device auto

echo ""
echo "Example 2: CVTG-2K Evaluation"
echo "-----------------------------"
python "$PROJECT_ROOT/eval/scripts/evaluate.py" \
    --mode generation \
    --benchmark "$PROJECT_ROOT/eval/CVTG-2K" \
    --benchmark_type cvtg \
    --generated "$PROJECT_ROOT/outputs/cvtg_results" \
    --output "$PROJECT_ROOT/eval_results/cvtg_results.json" \
    --metrics ocr clip \
    --device auto

echo ""
echo "Example 3: Using Config File"
echo "-----------------------------"
python "$PROJECT_ROOT/eval/scripts/evaluate.py" \
    --mode generation \
    --config "$PROJECT_ROOT/eval/configs/generation_config.yaml"

echo ""
echo "All examples completed!"
