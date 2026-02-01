#!/bin/bash
# Launch VLM reward servers on specified GPUs

MODEL="${MODEL:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-4,5,6,7}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

PIDS=()

cleanup() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    wait
}
trap cleanup EXIT

IFS=',' read -ra GPU_LIST <<< "$GPUS"

for i in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$i]}"
    p=$((PORT + i))
    log="$LOG_DIR/reward_gpu${gpu}.log"
    echo "Starting server on GPU $gpu, port $p"
    CUDA_VISIBLE_DEVICES=$gpu python train/rl_ip/reward_server.py \
        --model "$MODEL" --host "$HOST" --port "$p" --device cuda:0 >> "$log" 2>&1 &
    PIDS+=($!)
done

echo "Servers started: ${PIDS[*]}"
echo "Ctrl+C to stop"
wait
