#!/bin/bash

# Flux-Dev Text-to-Image Generation Script
# Usage: bash run.sh

python inference_fluxdev.py \
  --prompt "A beautiful landscape with mountains and a lake at sunset" \
  --output_path "output/fluxdev_demo.png" \
  --seed 42 \
  --steps 50 \
  --guidance_scale 7.5 \
  --height 1024 \
  --width 1024
