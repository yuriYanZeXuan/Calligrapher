#!/bin/bash
set -euo pipefail

# Evaluate planner-specific generations with decoupled VLM judges.
#
# Example:
#   BENCHMARK=UnseenWords ./rebuttal/run_vlm_decoupling_eval.sh \
#     --planner-dir qwen=/mnt/.../rebuttal/results/generation/qwen/UnseenWords \
#     --planner-dir gemini=/mnt/.../rebuttal/results/generation/gemini/UnseenWords \
#     --judge qwen --judge gemini

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

BENCHMARK="${BENCHMARK:-UnseenWords}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/rebuttal/results/vlm_decoupling/${BENCHMARK}}"
SAMPLE_SIZE="${SAMPLE_SIZE:-5}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

python rebuttal/evaluate_vlm_decoupling.py \
  --benchmark "${BENCHMARK}" \
  --output-dir "${OUTPUT_DIR}" \
  --sample-size "${SAMPLE_SIZE}" \
  --sample-seed "${SAMPLE_SEED}" \
  "$@"

echo "Rebuttal tables: ${OUTPUT_DIR}"
