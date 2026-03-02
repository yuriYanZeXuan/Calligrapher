# Z-Image
python ablation/qualitative_FD/run_zimage_wo_fd.py --seed 42

# QwenImage
python ablation/qualitative_FD/run_qwen_wo_fd.py --seed 42

# 不跑 Pass 3 风格化（只看 Pass 2 注入差异，速度更快）
python ablation/qualitative_FD/run_zimage_wo_fd.py --harmonizer none