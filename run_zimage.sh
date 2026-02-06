#!/bin/bash
# Z-Image 推理启动脚本
# 用法:
#   ./run_zimage.sh                    # 单 GPU
#   ./run_zimage.sh --tts              # 所有 GPU 并行 TTS
#   ./run_zimage.sh --tts --gpus 0,1,2,3  # 指定 GPU

cd "$(dirname "$0")"
python run_zimage.py "$@"
