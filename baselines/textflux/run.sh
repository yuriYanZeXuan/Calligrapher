#!/bin/bash

# --- Configuration ---

# Path to the pipeline model.
PIPELINE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux-fill"

# Path to the base transformer model.
TRANSFORMER_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux-fill/transformer"

# Path to the TextFlux LoRA weights.
LORA_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/textflux-lora"

# Path to the source image.
IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing/test1_source.png"

# Path to the mask image, defining the area for text inpainting.
MASK_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing/test1_mask.png"

# The text prompt for inpainting.
PROMPT="TextFlux"

# The path where the output image will be saved.
OUTPUT_PATH="output/textflux_example_test1.png"

# Random seed for reproducibility.
SEED=42

# Number of inference steps.
STEPS=50

# Guidance scale for the diffusion process.
GUIDANCE_SCALE=30.0

# --- Run the inference script ---

echo "Starting TextFlux inference..."

python inference_textflux.py \
    --pipeline_path "$PIPELINE_PATH" \
    --transformer_path "$TRANSFORMER_PATH" \
    --lora_path "$LORA_PATH" \
    --image_path "$IMAGE_PATH" \
    --mask_path "$MASK_PATH" \
    --prompt "$PROMPT" \
    --output_path "$OUTPUT_PATH" \
    --seed $SEED \
    --steps $STEPS \
    --guidance_scale $GUIDANCE_SCALE

echo "Inference complete. Image saved to $OUTPUT_PATH"
