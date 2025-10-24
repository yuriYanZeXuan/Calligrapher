#!/bin/bash

# GRPO 训练示例脚本
# 使用 GRPO 方法进行 RL 训练，适合快速实验和资源受限场景

# 确保在项目根目录运行
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo "Working directory: $(pwd)"

# 基本参数（根据实际服务器路径修改）
MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux-fill"
SIGLIP_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/siglip"
DATA_JSON="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing"
OUTPUT_DIR="./output_grpo"
REWARD_SERVER_URL="http://127.0.0.1:8000/score"

# (可选) 从预训练的 IP-Adapter 开始
# INITIAL_IP_ADAPTER_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/calligrapher/calligrapher.bin"

# 训练参数
RESOLUTION=512
BATCH_SIZE=1
GRADIENT_ACCUMULATION=2
LEARNING_RATE=1e-5
MAX_STEPS=10000

# RL 参数
RL_METHOD="grpo"
RL_WARMUP_STEPS=100
RL_NUM_IMAGES_PER_PROMPT=12
RL_NUM_BATCHES_PER_EPOCH=1
RL_NUM_INFERENCE_STEPS=20
RL_GUIDANCE_SCALE=3.5
RL_TIMESTEP_FRACTION=0.5
RL_NUM_INNER_EPOCHS=1

# GRPO 特定参数
RL_GRPO_CLIP_RANGE=0.2
RL_KL_BETA=0.1
RL_ADV_CLIP_MAX=5.0

# EMA 参数
USE_EMA="--use_ema"
EMA_DECAY=0.9999
EMA_UPDATE_INTERVAL=1

# Reward 参数
OCR_WEIGHT=0.7
VLM_WEIGHT=0.3

# 执行训练（使用模块方式运行，适合远程服务器）
python -m train.train \
    --model_type flux \
    --pretrained_model_name_or_path "$MODEL_PATH" \
    --siglip_path "$SIGLIP_PATH" \
    --train_data_json "$DATA_JSON" \
    --output_dir "$OUTPUT_DIR" \
    $([ -n "$INITIAL_IP_ADAPTER_PATH" ] && echo "--initial_ip_adapter_path=$INITIAL_IP_ADAPTER_PATH") \
    --resolution $RESOLUTION \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --max_train_steps $MAX_STEPS \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --offload_text_encoder_two \
    --use_rl \
    --rl_method $RL_METHOD \
    --rl_warmup_steps $RL_WARMUP_STEPS \
    --rl_num_images_per_prompt $RL_NUM_IMAGES_PER_PROMPT \
    --rl_num_batches_per_epoch $RL_NUM_BATCHES_PER_EPOCH \
    --rl_num_inference_steps $RL_NUM_INFERENCE_STEPS \
    --rl_guidance_scale $RL_GUIDANCE_SCALE \
    --rl_timestep_fraction $RL_TIMESTEP_FRACTION \
    --rl_num_inner_epochs $RL_NUM_INNER_EPOCHS \
    --rl_grpo_clip_range $RL_GRPO_CLIP_RANGE \
    --rl_kl_beta $RL_KL_BETA \
    --rl_adv_clip_max $RL_ADV_CLIP_MAX \
    --rl_per_prompt_stat_tracking \
    --reward_server_url "$REWARD_SERVER_URL" \
    --ocr_weight $OCR_WEIGHT \
    --vlm_weight $VLM_WEIGHT \
    $USE_EMA \
    --ema_decay $EMA_DECAY \
    --ema_update_interval $EMA_UPDATE_INTERVAL \
    --checkpointing_steps 500 \
    --report_to wandb \
    --logging_dir logs \
    --use_8bit_adam

