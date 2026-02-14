#!/bin/bash

# --- Configuration ---

# Path to the FluxText LoRA safetensors checkpoint.
MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/FLUX-Text/model_multisize/pytorch_lora_weights.safetensors"

# Path to the YAML config file.
CONFIG_PATH="train/config/word_512_size.yaml"

# The prompt describing what to inpaint in the masked area.
PROMPT='a sign on a wooden wall, that reads "AnyText"'

# The literal text to render in the masked region.
TEXT="AnyText"

# Path to the source image.
IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing/test1_source.png"

# Path to the mask image, defining the area for inpainting (white=text region).
MASK_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing/test1_mask.png"

# The path where the output image will be saved.
OUTPUT_PATH="output/fluxtext_inpaint_example.png"

# Random seed for reproducibility.
SEED=42

# Number of inference steps.
STEPS=28

# Guidance scale for the diffusion process.
GUIDANCE_SCALE=3.5

# --- Run the inference script ---

echo "Starting FluxText inpainting..."

python inference_fluxtext.py \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --prompt "$PROMPT" \
    --text "$TEXT" \
    --image_path "$IMAGE_PATH" \
    --mask_path "$MASK_PATH" \
    --output_path "$OUTPUT_PATH" \
    --seed $SEED \
    --steps $STEPS \
    --guidance_scale $GUIDANCE_SCALE

echo "Inference complete. Image saved to $OUTPUT_PATH"
