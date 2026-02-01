#!/bin/bash
# Launch pseudo reward servers (random scores, CPU only)

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
INSTANCES="${INSTANCES:-8}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

PIDS=()

cleanup() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    wait
}
trap cleanup EXIT

for ((i=0; i<INSTANCES; i++)); do
    p=$((PORT + i))
    log="$LOG_DIR/pseudo_$p.log"
    echo "Starting pseudo server on port $p"
    python train/rl_ip/pseudo_reward_server.py --host "$HOST" --port "$p" >> "$log" 2>&1 &
    PIDS+=($!)
done

echo "Servers started: ${PIDS[*]}"
echo "Ctrl+C to stop"
wait
