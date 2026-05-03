#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/sdv1-5}"
DIFFUSION_MODEL_PATH="${DIFFUSION_MODEL_PATH:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/textdiffuser2_inp}"
BENCHMARK="${BENCHMARK:-UnseenWords}"
GPUS="${GPUS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASELINES_DIR}/results/textdiffuser2/${BENCHMARK}}"

export PYTHONPATH="${BASELINES_DIR}/textdiffuser2/diffusers_td2/src:${BASELINES_DIR}/textdiffuser2:${BASELINES_DIR}:${BASELINES_DIR}/..:${PYTHONPATH:-}"
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1

# run_parallel_benchmark.py uses the default SD1.5 path from MODEL_PATHS for
# TextDiffuser-2's base model; override DIFFUSION_MODEL_PATH with --model_path.
python "${BASELINES_DIR}/run_parallel_benchmark.py" \
  --model textdiffuser2 \
  --model_path "${DIFFUSION_MODEL_PATH}" \
  --benchmark "${BENCHMARK}" \
  --gpus "${GPUS}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
