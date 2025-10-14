#!/bin/bash

# =========================================================================
# This script is a template for launching the IP-Adapter training.
# You need to fill in the placeholder paths before running.
# =========================================================================

# --- 1. Configuration ---

# -- Paths --
# Path to the base pretrained model (e.g., a FLUX or Qwen-Edit model from Hugging Face)
# PRETRAINED_MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/QwenEdit" 
PRETRAINED_MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux-fill" 

# Path to the SigLIP vision model (used by the IP-Adapter)
SIGLIP_MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/siglip" 

# Path to the VLM model (e.g., Qwen-VL-Chat) used for reward calculation
VLM_MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Qwen25VL-7B"

# Path to your training dataset directory
# This directory must contain 'self_bench.txt' and the image triplets.
TRAIN_DATA_DIR="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing"

# Directory where checkpoints and logs will be saved
OUTPUT_DIR="./output_qwen"

# -- Model Selection --
# Choose the model architecture you are training.
# Options: "flux" or "qwen"
MODEL_TYPE="flux"

# -- Training Mode --
# Set to "true" to enable GRPO-RL training after warmup steps.
# Set to "false" to run only supervised training.
USE_RL=true

# --- 2. Training Parameters ---

# -- Basic Parameters --
RESOLUTION=1024
TRAIN_BATCH_SIZE=1
NUM_TRAIN_EPOCHS=10
CHECKPOINTING_STEPS=500
LEARNING_RATE=1e-5

# -- Accelerator & Precision --
MIXED_PRECISION="fp16" # or "bf16" if your hardware supports it

# -- RL-Specific Parameters --
# Number of supervised steps before switching to RL
RL_WARMUP_STEPS=1000
# Weight for the OCR reward component
OCR_REWARD_WEIGHT=0.7
# Weight for the VLM score reward component
VLM_REWARD_WEIGHT=0.3


# --- 3. Launch Training ---

# This command uses `accelerate` to launch the training script.
# Make sure you have configured accelerate beforehand with `accelerate config`.

accelerate launch -m Calligrapher.train.train \
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
  --report_to="tensorboard" \
  $( [ "$USE_RL" = true ] && echo "--use_rl" ) \
  --rl_warmup_steps=$RL_WARMUP_STEPS \
  --ocr_weight=$OCR_REWARD_WEIGHT \
  --vlm_weight=$VLM_REWARD_WEIGHT \
  --vlm_model_path=$VLM_MODEL_PATH

echo "Training finished."
