#!/bin/bash
# =========================================================================
# Launch pseudo reward servers that return random scores (CPU only).
# Mimics the CLI of run_server.sh to ease switch-over during testing.
# =========================================================================

DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT=8000
DEFAULT_GPU_IDS=""
DEFAULT_INSTANCES=8
DEFAULT_SEED=""

HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
GPU_IDS="$DEFAULT_GPU_IDS"
INSTANCES_OVERRIDE=""
SEED="$DEFAULT_SEED"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --gpu-ids) GPU_IDS="$2"; shift ;;
        --instances) INSTANCES_OVERRIDE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --host        Host for the server. (Default: $DEFAULT_HOST)"
            echo "  --port        Base port for the server. (Default: $DEFAULT_PORT)"
            echo "  --gpu-ids     Comma-separated identifiers to control instance count (no GPUs used)."
            echo "  --instances   Number of pseudo servers to launch when --gpu-ids is omitted. (Default: $DEFAULT_INSTANCES)"
            echo "  --seed        Optional random seed base; increments per instance."
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

if [[ -n "$GPU_IDS" ]]; then
    IFS=',' read -r -a INSTANCES <<< "$GPU_IDS"
else
    COUNT=${INSTANCES_OVERRIDE:-$DEFAULT_INSTANCES}
    INSTANCES=()
    for ((i=0; i<COUNT; i++)); do
        INSTANCES+=("cpu-$i")
    done
fi
PIDS=()
LOG_FILES=()

cleanup() {
    echo "Cleaning up pseudo reward servers..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait
    echo "All pseudo reward servers stopped."
}

trap cleanup EXIT

CURRENT_PORT=$PORT
INDEX=0
for label in "${INSTANCES[@]}"; do
    LOG_FILE="${LOG_DIR}/reward_server_pseudo_${CURRENT_PORT}.log"
    echo "Launching pseudo server on $HOST:$CURRENT_PORT (log: $LOG_FILE)"
    INSTANCE_SEED=""
    if [[ -n "$SEED" ]]; then
        INSTANCE_SEED=$((SEED + INDEX))
    fi
    python train/rl_ip/pseudo_reward_server.py \
        --host "$HOST" \
        --port "$CURRENT_PORT" \
        ${INSTANCE_SEED:+--seed "$INSTANCE_SEED"} >> "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    LOG_FILES+=("$LOG_FILE")
    CURRENT_PORT=$((CURRENT_PORT + 1))
    INDEX=$((INDEX + 1))
done

echo "Pseudo reward server(s) started with PIDs: ${PIDS[*]}"
echo "Logs: ${LOG_FILES[*]}"
echo "Press Ctrl+C to stop."

wait
