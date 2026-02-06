#!/bin/bash
# Z-Image 推理启动脚本
# 用法:
#   ./run_zimage.sh                    # 单 GPU
#   ./run_zimage.sh --tts              # 所有 GPU 并行 TTS
python run_zimage.py --tts  # 指定 GPU
