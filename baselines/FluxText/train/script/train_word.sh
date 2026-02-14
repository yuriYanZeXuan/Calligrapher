export XFL_CONFIG=./train/config/word_multi_size.yaml

# Specify the WANDB API key
# export WANDB_API_KEY='YOUR_WANDB_API_KEY'

echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true

torchrun --nproc_per_node=16 -m src.train.train