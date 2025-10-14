#!/bin/bash
# ==============================================================================
# Simplified script to run the local data generation pipeline.
# It starts the necessary model servers, runs the pipeline, and then shuts
# down the servers automatically.
# ==============================================================================

# --- Configuration ---
# You can customize the GPU allocation and ports here.
IMAGE_SERVER_PORT=8000
IMAGE_SERVER_GPUS="0,1,2,3"

LLM_SERVER_PORT=8001
LLM_SERVER_GPUS="4,5,6,7"
LOG_FILE="server.log"

# --- Cleanup Function ---
# This function is called when the script exits to ensure background services are stopped.
cleanup() {
    echo -e "\nShutting down background services..."
    # This command kills all child processes started by this script.
    # It's a robust way to clean up without managing PID files.
    pkill -P $$
    echo "Services stopped."
}

# Trap signals to ensure cleanup runs even if the script is interrupted (e.g., with Ctrl+C)
trap cleanup EXIT SIGINT SIGTERM

# --- Main Execution ---

echo "Starting local model services in the background..."
# Clear the log file for a fresh run
> $LOG_FILE

# Start Image Service, redirecting its output to the log file
echo "-> Launching Image Service on port $IMAGE_SERVER_PORT (GPUs: $IMAGE_SERVER_GPUS)..."
python local_server.py --service image --port $IMAGE_SERVER_PORT --device_ids $IMAGE_SERVER_GPUS >> $LOG_FILE 2>&1 &

# Start LLM Service, redirecting its output to the log file
echo "-> Launching LLM Service on port $LLM_SERVER_PORT (GPUs: $LLM_SERVER_GPUS)..."
python local_server.py --service llm --port $LLM_SERVER_PORT --device_ids $LLM_SERVER_GPUS >> $LOG_FILE 2>&1 &

echo "Service logs are being written to '$LOG_FILE'. You can tail it in another terminal."
echo -e "\nWaiting for models to load. This can take several minutes..."
# We use a simple sleep timer. If your models load faster or slower, you can
# adjust this value (in seconds).
sleep 120

echo -e "\n----------------------------------------"
echo "Services are assumed to be ready. Running the main pipeline..."
echo "----------------------------------------\n"
python main.py --service local

echo -e "\n----------------------------------------"
echo "Pipeline execution finished."
echo "----------------------------------------"

# The cleanup function will be called automatically when the script exits.
exit 0
