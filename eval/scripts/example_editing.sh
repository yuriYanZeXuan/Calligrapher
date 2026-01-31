#!/bin/bash
# Example: Editing Evaluation Scripts
# 
# These are example commands for running editing evaluation.
# Modify the paths according to your setup.

set -e

# Configuration
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"

echo "==================================="
echo "Editing Evaluation Examples"
echo "==================================="
echo ""

# Example 1: Directory-based benchmark (like Calligrapher_bench_testing)
echo "Example 1: Directory-based Editing Evaluation"
echo "----------------------------------------------"
python "$PROJECT_ROOT/eval/scripts/evaluate.py" \
    --mode editing \
    --benchmark "$PROJECT_ROOT/eval/Calligrapher_bench_testing" \
    --benchmark_format directory \
    --generated "$PROJECT_ROOT/outputs/editing_results" \
    --output "$PROJECT_ROOT/eval_results/editing_results.json" \
    --metrics ocr dino clip \
    --device auto

echo ""
echo "Example 2: TXT-based Benchmark"
echo "-------------------------------"
python "$PROJECT_ROOT/eval/scripts/evaluate.py" \
    --mode editing \
    --benchmark "$PROJECT_ROOT/eval/benchmark.txt" \
    --benchmark_format txt \
    --generated "$PROJECT_ROOT/outputs/editing_results" \
    --output "$PROJECT_ROOT/eval_results/editing_txt_results.json" \
    --metrics ocr dino \
    --use_masked_metrics \
    --device auto

echo ""
echo "Example 3: Using Config File"
echo "-----------------------------"
python "$PROJECT_ROOT/eval/scripts/evaluate.py" \
    --mode editing \
    --config "$PROJECT_ROOT/eval/configs/editing_config.yaml"

echo ""
echo "All examples completed!"
