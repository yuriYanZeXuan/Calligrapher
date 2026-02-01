
export CUDA_VISIBLE_DEVICES=0

PRETRAINED_MODEL="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-image-sft"
TRAIN_DATA_JSON="samples/LTB_dataset.jsonl"
OUTPUT_DIR="output/zimage_lora_run"

# Make sure `peft` is installed in your environment:
#   pip install peft

accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  --machine_rank 0 \
  --main_process_port 29500 \
  -m train.train_gen \
  --model_type zimage \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
  --train_data_json "${TRAIN_DATA_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_train_steps 10000 \
  --checkpointing_steps 500 \
  --max_sequence_length 512 \
  --use_zimage_lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.0

