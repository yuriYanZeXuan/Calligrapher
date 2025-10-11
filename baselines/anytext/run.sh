#!/bin/bash

# --- Configuration ---

# Path to the directory containing the AnyText2 models.
MODEL_DIR="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/anytext2"

# The general prompt describing the image.
IMG_PROMPT="A sign on a wooden wall"

# The text to be inpainted into the masked area.
# Use quotes to enclose the text content.
TEXT_PROMPT='"AnyText"'

# Paths to your source image and mask image.
# These paths are examples, please replace them with your actual file paths.
SOURCE_IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing/test1_source.png"
MASK_IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing/test1_mask.png"

# Directory where the output image will be saved.
OUTPUT_DIR="output/anytext_inpaint_example"

# Random seed for reproducibility.
SEED=42

# --- Run the inference script ---

echo "Starting AnyText2 inpainting..."

python inference_anytext.py \
    --mode "inpaint" \
    --model_dir "$MODEL_DIR" \
    --img_prompt "$IMG_PROMPT" \
    --text_prompt "$TEXT_PROMPT" \
    --source_path "$SOURCE_IMAGE_PATH" \
    --mask_path "$MASK_IMAGE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --seed $SEED

echo "Inpainting complete. Image saved to $OUTPUT_DIR"
