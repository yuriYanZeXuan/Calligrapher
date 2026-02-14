#!/bin/bash

# --- Configuration ---

# Path to the FluxText LoRA safetensors checkpoint.
MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/fluxtext_lora.safetensors"

# Path to the YAML config file.
CONFIG_PATH="train/config/word_512_size.yaml"

# The prompt describing the desired image (text to render is quoted inside).
PROMPT='a logo that reads "HELLO WORLD"'

# Optional: explicit text to render (overrides parsing from prompt).
# TEXT="HELLO WORLD"

# Optional: background image path (defaults to a white canvas).
# IMAGE_PATH="/path/to/background.png"

# The path where the output image will be saved.
OUTPUT_PATH="output/fluxtext_example.png"

# Random seed for reproducibility.
SEED=42

# Number of inference steps.
STEPS=28

# Guidance scale for the diffusion process.
GUIDANCE_SCALE=3.5

# Image dimensions.
WIDTH=512
HEIGHT=512

# --- Run the inference script ---

echo "Starting FluxText inference..."

python inference_fluxtext.py \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --prompt "$PROMPT" \
    --output_path "$OUTPUT_PATH" \
    --seed $SEED \
    --steps $STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --width $WIDTH \
    --height $HEIGHT

echo "Inference complete. Image saved to $OUTPUT_PATH"
