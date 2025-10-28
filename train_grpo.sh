#!/bin/bash

# GRPO 训练示例脚本
# 使用 GRPO 方法进行 RL 训练，适合快速实验和资源受限场景

# =========================================================================
# 基本配置（参考 run_training.sh）
# =========================================================================

# 基本参数
MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux-fill"
SIGLIP_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/siglip"
DATA_JSON="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/dataset/Calligrapher_bench_testing"
OUTPUT_DIR="./output_grpo"

# (可选) 从预训练的 IP-Adapter 开始
# INITIAL_IP_ADAPTER_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/calligrapher/calligrapher.bin"

# Reward Server 配置（需与 run_server.sh 匹配）
REWARD_SERVER_HOST="127.0.0.1"
REWARD_SERVER_PORT=8000

# 硬件配置
# 训练 GPU（例如 "0,1,2,3"，如果为空则使用单 GPU）
TRAINING_GPU_IDS="0,1,2,3"

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

# =========================================================================
# 构建 Reward Server URLs（与 run_training.sh 逻辑一致）
# =========================================================================

# 计算进程数
ACCELERATE_LAUNCH_ARGS=""
NUM_PROCESSES=1
if [ -n "$TRAINING_GPU_IDS" ]; then
    NUM_PROCESSES=$(echo "$TRAINING_GPU_IDS" | awk -F',' '{print NF}')
    ACCELERATE_LAUNCH_ARGS="--num_processes=$NUM_PROCESSES --gpu_ids=$TRAINING_GPU_IDS"
fi

# 构建多个 reward server URLs（每个训练进程连接一个 server）
REWARD_SERVER_URLS=""
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    port=$((REWARD_SERVER_PORT + i))
    url="http://${REWARD_SERVER_HOST}:${port}/score"
    if [ -z "$REWARD_SERVER_URLS" ]; then
        REWARD_SERVER_URLS="$url"
    else
        REWARD_SERVER_URLS="$REWARD_SERVER_URLS,$url"
    fi
done

echo "Training will run on GPU(s): ${TRAINING_GPU_IDS:-"Default GPU"}"
echo "Connecting to Reward Server(s) at: $REWARD_SERVER_URLS"
echo ""

# =========================================================================
# 执行训练
# =========================================================================

accelerate launch $ACCELERATE_LAUNCH_ARGS -m train.train \
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
    --reward_server_url "$REWARD_SERVER_URLS" \
    --ocr_weight $OCR_WEIGHT \
    --vlm_weight $VLM_WEIGHT \
    $USE_EMA \
    --ema_decay $EMA_DECAY \
    --ema_update_interval $EMA_UPDATE_INTERVAL \
    --checkpointing_steps 500 \
    --report_to wandb \
    --logging_dir logs \
    --use_8bit_adam

