#!/bin/bash

# --- Configuration ---

# The model path for Qwen-Image-Edit.
# You can use the default from Hugging Face or a local path if you have it downloaded.
MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/QwenEdit"

# The editing instruction for the image.
PROMPT="Edit text to 'World'"

# Path to the source image that you want to edit.
# Please replace this with the actual path to your image.
IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing/test1_source.png"

# The path where the edited image will be saved.
OUTPUT_PATH="output/qwenedit_example_test1.png"

# Random seed for reproducibility.
SEED=42

# Number of inference steps.
STEPS=50

# --- Run the inference script ---

echo "Starting Qwen-Image-Edit inference..."

python inference_qwenedit.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_path "$IMAGE_PATH" \
    --output_path "$OUTPUT_PATH" \
    --seed $SEED \
    --steps $STEPS

echo "Inference complete. Image saved to $OUTPUT_PATH"
