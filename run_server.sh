#!/bin/bash
# =========================================================================
# This script launches the VLM reward model server.
# It can launch multiple server processes on specified GPUs, each on a
# subsequent port, and runs in the foreground to display logs.
# =========================================================================

# --- Configuration ---
# Default values can be overridden by command-line arguments.
DEFAULT_VLM_MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT=8000
DEFAULT_GPU_IDS="4,5,6,7"

# --- Argument Parsing ---
VLM_MODEL_PATH="$DEFAULT_VLM_MODEL_PATH"
HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
GPU_IDS="$DEFAULT_GPU_IDS"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --vlm-model-path) VLM_MODEL_PATH="$2"; shift ;;
        --host) HOST="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --gpu-ids) GPU_IDS="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --vlm-model-path  Path to the VLM reward model. (Default: $DEFAULT_VLM_MODEL_PATH)"
            echo "  --host            Host for the server. (Default: $DEFAULT_HOST)"
            echo "  --port            Base port for the server. (Default: $DEFAULT_PORT)"
            echo "  --gpu-ids         Comma-separated list of GPU IDs to run servers on (e.g., '0,1,7'). (Default: $DEFAULT_GPU_IDS)"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Launch Server(s) ---
echo "--- Launching Reward Server(s) ---"
echo "Model Path: $VLM_MODEL_PATH"
echo "Base Listening Port: $PORT"
echo "Running on GPU ID(s): $GPU_IDS"
echo "---------------------------------"

# Convert comma-separated string to an array of GPUs
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
PIDS=()

# Define a function to clean up all server processes on exit
cleanup() {
    echo "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        echo "Stopping server (PID: $pid)..."
        # Use a gentle kill signal first, then a stronger one if needed
        kill "$pid" 2>/dev/null
    done
    # Wait for all background processes to terminate
    wait
    echo "All reward servers stopped."
}

# Trap EXIT signal to call the cleanup function, ensuring servers are killed
# when the script is terminated (e.g., with Ctrl+C).
trap cleanup EXIT

CURRENT_PORT=$PORT
for gpu_id in "${GPUS[@]}"; do
  echo "Launching server on GPU $gpu_id at $HOST:$CURRENT_PORT..."
  
  # Set CUDA_VISIBLE_DEVICES to isolate the GPU for the process.
  # The python script can then use "cuda:0" to refer to this isolated GPU.
  CUDA_VISIBLE_DEVICES=$gpu_id python train/rl_ip/reward_server.py \
    --model_path "$VLM_MODEL_PATH" \
    --host "$HOST" \
    --port "$CURRENT_PORT" \
    --device "cuda:0" &
  
  PIDS+=($!)
  CURRENT_PORT=$((CURRENT_PORT + 1))
done

echo "Reward server(s) started with PIDs: ${PIDS[*]}"
echo "Press Ctrl+C to stop all servers."

# Wait for all background PIDs to finish. The `trap` will handle cleanup.
wait
