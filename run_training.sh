#!/bin/bash
export PYTHONPATH=$(pwd)

# =========================================================================
# This script is for launching the IP-Adapter training.
# It connects to a separate reward server for RL-based training.
# =========================================================================

# --- 1. Configuration ---

# -- Paths --
PRETRAINED_MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux-fill" 
SIGLIP_MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/siglip" 
TRAIN_DATA_DIR="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing"
OUTPUT_DIR="./output_flux"
# --- Path to optionally initialize IP-Adapter from ---
INITIAL_IP_ADAPTER_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/calligrapher/calligrapher.bin" # e.g., "/path/to/flux_ip/output/checkpoint-XXXX/ip-adapter.bin"

# -- Model Selection --
MODEL_TYPE="flux" # "flux" or "qwen"

# -- Training Mode --
USE_RL=true
USE_8BIT_ADAM=true
OFFLOAD_TEXT_ENCODER_TWO=true
# Set to "true" to disable connecting to the reward server and use random rewards instead.
DISABLE_RL_REWARD_MODEL=false

# -- Hardware Configuration --
# Specify the GPU IDs to use for training, e.g., "0,1,2,3".
# If empty, `accelerate` will use its default configuration from `accelerate config`.
TRAINING_GPU_IDS="0,1,2,3"
GRADIENT_ACCUMULATION_STEPS=2
GRADIENT_CHECKPOINTING=true

# --- 2. Reward Server Configuration ---
# These should match the host and port of your running reward server.
REWARD_SERVER_HOST="127.0.0.1"
REWARD_SERVER_PORT=8000


# --- 3. Training Parameters ---

# -- Basic Parameters --
RESOLUTION=512
TRAIN_BATCH_SIZE=1
NUM_TRAIN_EPOCHS=10
CHECKPOINTING_STEPS=500
LEARNING_RATE=1e-5

# -- Accelerator & Precision --
MIXED_PRECISION="fp16" # or "bf16"

# -- RL-Specific Parameters --
RL_WARMUP_STEPS=100
OCR_REWARD_WEIGHT=0.7
VLM_REWARD_WEIGHT=0.3

# -- New RL Sampling and Training Parameters --
RL_NUM_BATCHES_PER_EPOCH=1
RL_NUM_IMAGES_PER_PROMPT=12
RL_NUM_INFERENCE_STEPS=20
RL_TIMESTEP_FRACTION=0.5
RL_NUM_INNER_EPOCHS=1
RL_GRPO_CLIP_RANGE=0.2
RL_KL_BETA=0.1


# --- 4. Launch Training ---
# This script uses `accelerate launch` for training.
#
# The `TRAINING_GPU_IDS` variable above allows you to specify which GPUs to use.
# If you leave it empty, you must configure `accelerate` beforehand by running
# `accelerate config` in your terminal.

echo "--- Starting Training ---"

# Override server settings with command-line arguments if provided
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reward-host) REWARD_SERVER_HOST="$2"; shift ;;
        --reward-port) REWARD_SERVER_PORT="$2"; shift ;;
        --gpu-ids) TRAINING_GPU_IDS="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --reward-host   Host of the reward server. (Default: $REWARD_SERVER_HOST)"
            echo "  --reward-port   Port of the reward server. (Default: $REWARD_SERVER_PORT)"
            echo "  --gpu-ids       Comma-separated GPU IDs for training, e.g., '0,1,2'. (Default: $TRAINING_GPU_IDS)"
            echo "  Note: Other parameters like model paths must be configured inside the script."
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Training will run on GPU(s): ${TRAINING_GPU_IDS:-"Accelerate Default"}"

ACCELERATE_LAUNCH_ARGS=""
NUM_PROCESSES=1
if [ -n "$TRAINING_GPU_IDS" ]; then
  NUM_PROCESSES=$(echo "$TRAINING_GPU_IDS" | awk -F',' '{print NF}')
  ACCELERATE_LAUNCH_ARGS="--num_processes=$NUM_PROCESSES --gpu_ids=$TRAINING_GPU_IDS"
fi

REWARD_SERVER_URLS=""
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    port=$((REWARD_SERVER_PORT + i))
    url="http://${REWARD_SERVER_HOST}:${port}/score"
    if [ -z "$REWARD_SERVER_URLS" ]; then
        REWARD_SERVER_URLS="$url"
    else
        REWARD_SERVER_URLS="$REWARD_SERVER_URLS,$url"
    fi
done

echo "Connecting to Reward Server(s) at: $REWARD_SERVER_URLS"

export CUDA_LAUNCH_BLOCKING=1

accelerate launch $ACCELERATE_LAUNCH_ARGS train/train.py \
  --pretrained_model_name_or_path=$PRETRAINED_MODEL_PATH \
  --siglip_path=$SIGLIP_MODEL_PATH \
  --train_data_json=$TRAIN_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --model_type=$MODEL_TYPE \
  --resolution=$RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --learning_rate=$LEARNING_RATE \
  --mixed_precision=$MIXED_PRECISION \
  --report_to="wandb" \
  $( [ -n "$INITIAL_IP_ADAPTER_PATH" ] && echo "--initial_ip_adapter_path=$INITIAL_IP_ADAPTER_PATH" ) \
  $( [ "$USE_RL" = true ] && echo "--use_rl" ) \
  --rl_warmup_steps=$RL_WARMUP_STEPS \
  --ocr_weight=$OCR_REWARD_WEIGHT \
  --vlm_weight=$VLM_REWARD_WEIGHT \
  --reward_server_url="$REWARD_SERVER_URLS" \
  $( [ "$DISABLE_RL_REWARD_MODEL" = true ] && echo "--no_rl_reward_model" ) \
  $( [ "$USE_8BIT_ADAM" = true ] && echo "--use_8bit_adam" )\
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  $( [ "$GRADIENT_CHECKPOINTING" = true ] && echo "--gradient_checkpointing" ) \
  $( [ "$OFFLOAD_TEXT_ENCODER_TWO" = true ] && echo "--offload_text_encoder_two" ) \
  --enable_memory_profiler \
  --rl_per_prompt_stat_tracking \
  --rl_num_batches_per_epoch=$RL_NUM_BATCHES_PER_EPOCH \
  --rl_num_images_per_prompt=$RL_NUM_IMAGES_PER_PROMPT \
  --rl_num_inference_steps=$RL_NUM_INFERENCE_STEPS \
  --rl_timestep_fraction=$RL_TIMESTEP_FRACTION \
  --rl_num_inner_epochs=$RL_NUM_INNER_EPOCHS \
  --rl_grpo_clip_range=$RL_GRPO_CLIP_RANGE \
  --rl_kl_beta=$RL_KL_BETA

echo "Training finished."
