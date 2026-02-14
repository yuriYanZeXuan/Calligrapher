torchrun --nproc_per_node 8 eval/anytext_inference.py \
        --config_path model_path/config.yaml \
        --model_path model_path/pytorch_lora_weights.safetensors \
        --json_path AnyText-benchmark/benchmark/wukong_word/test1k_valid.json \
        --output_dir wukong_word

