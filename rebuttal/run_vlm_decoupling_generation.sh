#!/bin/bash
set -euo pipefail

# Generate rebuttal samples with a selected planner VLM.
#
# Examples:
#   PLANNER=qwen BENCHMARK=UnseenWords GPUS=8 ./rebuttal/run_vlm_decoupling_generation.sh --debug
#   PLANNER=gemini GLYPH_PLANNER_VLM_BASE_URL=http://127.0.0.1:51958/v1 ./rebuttal/run_vlm_decoupling_generation.sh
#
# For Gemini, start Paper2Slides/gemini_proxy.py first. API keys are read from
# Calligrapher/.env or the process environment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

PLANNER="${PLANNER:-qwen}"
BENCHMARK="${BENCHMARK:-UnseenWords}"
GPUS="${GPUS:-8}"
MODEL="${MODEL:-ours_qwen}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/rebuttal/results/generation/${PLANNER}/${BENCHMARK}}"
SAMPLE_SIZE="${SAMPLE_SIZE:-5}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

case "${PLANNER}" in
  qwen|qwen3|qwen3-vl)
    export GLYPH_PLANNER_VLM_MODEL="${GLYPH_PLANNER_VLM_MODEL:-qwen3-vl-235b-a22b-instruct}"
    export GLYPH_PLANNER_VLM_BASE_URL="${GLYPH_PLANNER_VLM_BASE_URL:-${QST_BASE_URL:-}}"
    export GLYPH_PLANNER_VLM_API_KEY="${GLYPH_PLANNER_VLM_API_KEY:-${QST_API_KEY:-}}"
    export GLYPH_PLANNER_VLM_API_KEY2="${GLYPH_PLANNER_VLM_API_KEY2:-${QST_API_KEY2:-}}"
    ;;
  gemini|gemini3|gemini-3-pro)
    export GLYPH_PLANNER_VLM_MODEL="${GLYPH_PLANNER_VLM_MODEL:-gemini-3-pro}"
    export GLYPH_PLANNER_VLM_BASE_URL="${GLYPH_PLANNER_VLM_BASE_URL:-${GEMINI_PROXY_BASE_URL:-http://127.0.0.1:51958/v1}}"
    export GLYPH_PLANNER_VLM_API_KEY="${GLYPH_PLANNER_VLM_API_KEY:-${GEMINI3_API_KEY:-}}"
    unset GLYPH_PLANNER_VLM_API_KEY2 || true
    ;;
  gpt4o|gpt-4o)
    export GLYPH_PLANNER_VLM_MODEL="${GLYPH_PLANNER_VLM_MODEL:-gpt-4o}"
    export GLYPH_PLANNER_VLM_BASE_URL="${GLYPH_PLANNER_VLM_BASE_URL:-${GPT4O_BASE_URL:-https://runway.devops.rednote.life/openai}}"
    export GLYPH_PLANNER_VLM_API_KEY="${GLYPH_PLANNER_VLM_API_KEY:-${GPT4O_API_KEY:-}}"
    unset GLYPH_PLANNER_VLM_API_KEY2 || true
    ;;
  *)
    echo "Unknown PLANNER=${PLANNER}. Set GLYPH_PLANNER_VLM_* manually for custom backends." >&2
    ;;
esac

mkdir -p "${OUTPUT_DIR}"

python baselines/run_parallel_benchmark.py \
  --model "${MODEL}" \
  --benchmark "${BENCHMARK}" \
  --gpus "${GPUS}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample-size "${SAMPLE_SIZE}" \
  --sample-seed "${SAMPLE_SEED}" \
  --skip-eval \
  "$@"

echo "Generated images: ${OUTPUT_DIR}"
