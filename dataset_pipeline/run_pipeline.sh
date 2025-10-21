#!/bin/bash
# ==============================================================================
# Script to run the data generation pipeline with multiple parallel service pairs.
# It launches 4 independent pairs of (Image Service + LLM Service), each on
# dedicated GPUs, and then runs a parallel client to communicate with them.
# ==============================================================================

# --- Configuration ---
NUM_PAIRS=4
IMAGE_BASE_PORT=8000
LLM_BASE_PORT=9000
LOG_FILE="server.log"
NUM_WORKERS=4 # Number of parallel client workers

# --- Cleanup Function ---
cleanup() {
    echo -e "\nShutting down background services..."
    pkill -P $$
    echo "Services stopped."
}

trap cleanup EXIT SIGINT SIGTERM

# --- Main Execution ---

echo "Starting $NUM_PAIRS local model service pairs in the background..."
> $LOG_FILE # Clear the log file

IMAGE_PORTS=()
LLM_PORTS=()

for i in $(seq 0 $(($NUM_PAIRS - 1)))
do
    LLM_GPU_ID=$(($i * 2))
    IMAGE_GPU_ID=$(($i * 2 + 1))
    
    IMAGE_PORT=$(($IMAGE_BASE_PORT + $i))
    LLM_PORT=$(($LLM_BASE_PORT + $i))

    IMAGE_PORTS+=($IMAGE_PORT)
    LLM_PORTS+=($LLM_PORT)

    echo "-> Launching Pair $i: Image Service on GPU $IMAGE_GPU_ID (Port: $IMAGE_PORT) and LLM Service on GPU $LLM_GPU_ID (Port: $LLM_PORT)..."

    # Start Image Service for the pair
    python local_server.py --service image --port $IMAGE_PORT --device_ids $IMAGE_GPU_ID >> $LOG_FILE 2>&1 &
    
    # Start LLM Service for the pair
    python local_server.py --service llm --port $LLM_PORT --device_ids $LLM_GPU_ID >> $LOG_FILE 2>&1 &
done

echo "Service logs are being written to '$LOG_FILE'."
echo -e "\nWaiting for models to load. This can take several minutes..."
sleep 180 # Increased wait time for multiple model loading

# Convert arrays to comma-separated strings for passing as arguments
IMAGE_PORTS_STR=$(IFS=,; echo "${IMAGE_PORTS[*]}")
LLM_PORTS_STR=$(IFS=,; echo "${LLM_PORTS[*]}")

echo -e "\n----------------------------------------"
echo "Services are assumed to be ready. Running the main pipeline with $NUM_WORKERS workers across $NUM_PAIRS service pairs..."
echo "Image Ports: $IMAGE_PORTS_STR | LLM Ports: $LLM_PORTS_STR"
echo "----------------------------------------\n"

python main.py --service local \
               --num-workers $NUM_WORKERS \
               --image-ports $IMAGE_PORTS_STR \
               --llm-ports $LLM_PORTS_STR \
               --instructions-file /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/dataset_pipeline/instructions_20000_generated.txt

echo -e "\n----------------------------------------"
echo "Pipeline execution finished."
echo "----------------------------------------"

exit 0
