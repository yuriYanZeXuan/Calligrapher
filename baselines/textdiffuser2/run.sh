#!/bin/bash

BASE_MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/sdv1-5"

DIFFUSION_MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/textdiffuser2_inp"

PROMPT="Hello World"

SOURCE_IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing/test1_source.png"

MASK_IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing/test1_mask.png"

OUTPUT_PATH="output/example_inpaint.png"

# --- Run the inference script ---

echo "Starting TextDiffuser-2 inpainting..."

python inference_textdiffuser2.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --diffusion_model_path "$DIFFUSION_MODEL_PATH" \
    --prompt "$PROMPT" \
    --source_path "$SOURCE_IMAGE_PATH" \
    --mask_path "$MASK_IMAGE_PATH" \
    --output_path "$OUTPUT_PATH" \
    --seed 42

echo "Inpainting complete. Image saved to $OUTPUT_PATH"
