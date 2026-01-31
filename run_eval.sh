#!/bin/bash
# Calligrapher Unified Evaluation Script
# 
# Usage:
#   ./run_eval.sh generation --config eval/configs/generation_config.yaml
#   ./run_eval.sh editing --config eval/configs/editing_config.yaml
#   ./run_eval.sh generation --benchmark eval/OneIG-Bench/OneIG-Bench.json --generated outputs/results
#
# Modes:
#   generation - Evaluate text-to-image generation
#   editing    - Evaluate text rendering editing with masks

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <mode> [options]"
    echo ""
    echo "Modes:"
    echo "  generation  - Evaluate text-to-image generation"
    echo "  editing     - Evaluate text rendering editing"
    echo ""
    echo "Examples:"
    echo "  $0 generation --config eval/configs/generation_config.yaml"
    echo "  $0 editing --config eval/configs/editing_config.yaml"
    echo "  $0 generation --benchmark eval/OneIG-Bench/OneIG-Bench.json --generated outputs/results"
    exit 1
fi

MODE=$1
shift

# Run evaluation
python eval/scripts/evaluate.py --mode "$MODE" "$@"
