#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen-image-2512}"
BENCHMARK="${BENCHMARK:-UnseenWords}"
GPUS="${GPUS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASELINES_DIR}/results/freetext_qwen/${BENCHMARK}}"

export PYTHONPATH="${BASELINES_DIR}/FreeText:${BASELINES_DIR}:${BASELINES_DIR}/..:${PYTHONPATH:-}"
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1

python "${BASELINES_DIR}/run_parallel_benchmark.py" \
  --model freetext_qwen \
  --model_path "${MODEL_PATH}" \
  --benchmark "${BENCHMARK}" \
  --gpus "${GPUS}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
