#!/bin/bash

# --- Configuration ---

# Path to the Flux-Fill model.
# You can use the default from Hugging Face or a local path if you have it downloaded.
MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux-fill"

# The prompt describing what to inpaint in the masked area.
PROMPT="Edit text to 'World'"

# Path to the source image.
IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing/test1_source.png"

# Path to the mask image, defining the area for inpainting.
MASK_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing/test1_mask.png"

# The path where the output image will be saved.
OUTPUT_PATH="output/fluxfill_example_test1.png"

# Random seed for reproducibility.
SEED=42

# Number of inference steps.
STEPS=50

# Guidance scale for the diffusion process.
GUIDANCE_SCALE=30.0

# --- Run the inference script ---

echo "Starting Flux-Fill inference..."

python inference_fluxfill.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_path "$IMAGE_PATH" \
    --mask_path "$MASK_PATH" \
    --output_path "$OUTPUT_PATH" \
    --seed $SEED \
    --steps $STEPS \
    --guidance_scale $GUIDANCE_SCALE

echo "Inference complete. Image saved to $OUTPUT_PATH"
