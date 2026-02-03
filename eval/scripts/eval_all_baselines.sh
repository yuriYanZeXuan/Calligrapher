# bash eval/scripts/eval_all_baselines.sh
MINERU_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/MinerU_VLM"
VLM_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B"

MODELS=(qwenimage zimage glmimage nanobanana fluxdev fluxklein)

BENCH_NAMES=(LongText-Bench CVTG-2K OneIG-Bench)
BENCH_PATHS=(eval/LongText-Bench eval/CVTG-2K eval/OneIG-Bench/OneIG-Bench_text.json)
BENCH_TYPES=(longtext cvtg oneig)

OUT_DIR="eval_results/baselines_all"
mkdir -p "${OUT_DIR}"

for model in "${MODELS[@]}"; do
  for i in "${!BENCH_NAMES[@]}"; do
    bench="${BENCH_NAMES[$i]}"
    bench_path="${BENCH_PATHS[$i]}"
    bench_type="${BENCH_TYPES[$i]}"
    gen_dir="baselines/results/${model}/${bench}"
    out="${OUT_DIR}/${model}_${bench}.jsonl"

    python3 eval/scripts/eval_parallel.py \
      --results_dir "${gen_dir}" \
      --benchmark "${bench_path}" \
      --benchmark_type "${bench_type}" \
      --output "${out}" \
      --gpus 8 \
      --metrics vqa ocr clip vlm aesthetic \
      --mineru_path "${MINERU_PATH}" \
      --vlm_path "${VLM_PATH}" \
      --resume \
  done
done