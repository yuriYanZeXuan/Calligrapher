#!/bin/bash
set -euo pipefail

# One-click small-scale VLM decoupling experiment for rebuttal.
# It runs:
#   1. generation with Qwen planner
#   2. generation with Gemini planner
#   3. evaluation with Qwen and Gemini judges
#
# Defaults: 5 random samples from UnseenWords, seed 42, 1 GPU.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

BENCHMARK="${BENCHMARK:-UnseenWords}"
SAMPLE_SIZE="${SAMPLE_SIZE:-5}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
GPUS="${GPUS:-1}"
MODEL="${MODEL:-ours_qwen}"
SKIP_GEMINI="${SKIP_GEMINI:-0}"

QWEN_DIR="${ROOT_DIR}/rebuttal/results/generation/qwen/${BENCHMARK}"
GEMINI_DIR="${ROOT_DIR}/rebuttal/results/generation/gemini/${BENCHMARK}"
EVAL_DIR="${ROOT_DIR}/rebuttal/results/vlm_decoupling/${BENCHMARK}"

echo "[1/3] Generate with Qwen planner -> ${QWEN_DIR}"
PLANNER=qwen BENCHMARK="${BENCHMARK}" SAMPLE_SIZE="${SAMPLE_SIZE}" SAMPLE_SEED="${SAMPLE_SEED}" \
  GPUS="${GPUS}" MODEL="${MODEL}" OUTPUT_DIR="${QWEN_DIR}" \
  ./rebuttal/run_vlm_decoupling_generation.sh --resume

PLANNER_DIR_ARGS=(--planner-dir "qwen=${QWEN_DIR}")
JUDGE_ARGS=(--judge qwen)

if [ "${SKIP_GEMINI}" != "1" ]; then
  echo "[2/3] Generate with Gemini planner -> ${GEMINI_DIR}"
  echo "      If this fails, start Paper2Slides/gemini_proxy.py or set SKIP_GEMINI=1."
  PLANNER=gemini BENCHMARK="${BENCHMARK}" SAMPLE_SIZE="${SAMPLE_SIZE}" SAMPLE_SEED="${SAMPLE_SEED}" \
    GPUS="${GPUS}" MODEL="${MODEL}" OUTPUT_DIR="${GEMINI_DIR}" \
    ./rebuttal/run_vlm_decoupling_generation.sh --resume
  PLANNER_DIR_ARGS+=(--planner-dir "gemini=${GEMINI_DIR}")
  JUDGE_ARGS+=(--judge gemini)
else
  echo "[2/3] Skip Gemini planner/judge (SKIP_GEMINI=1)"
fi

echo "[3/3] Cross-evaluate and write rebuttal tables -> ${EVAL_DIR}"
BENCHMARK="${BENCHMARK}" SAMPLE_SIZE="${SAMPLE_SIZE}" SAMPLE_SEED="${SAMPLE_SEED}" OUTPUT_DIR="${EVAL_DIR}" \
  ./rebuttal/run_vlm_decoupling_eval.sh \
  "${PLANNER_DIR_ARGS[@]}" \
  "${JUDGE_ARGS[@]}"

echo
echo "Done."
echo "Images:"
echo "  Qwen planner:   ${QWEN_DIR}"
if [ "${SKIP_GEMINI}" != "1" ]; then
  echo "  Gemini planner: ${GEMINI_DIR}"
fi
echo "Tables:"
echo "  ${EVAL_DIR}/vlm_decoupling_summary.md"
echo "  ${EVAL_DIR}/vlm_decoupling_summary.csv"
echo "  ${EVAL_DIR}/vlm_decoupling_details.csv"
