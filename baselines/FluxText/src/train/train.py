from datetime import timedelta
import os
import time

from datasets import load_dataset
import lightning as L
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import yaml

from .callbacks import TrainingCallback
from ..data.dataloader import prepare_dataloader
from .data_word import T3DataSetWarp
from .model import OminiModelFIll
from .parallel_states import get_data_parallel_group


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_device():
    """
    Setup the device and the distributed coordinator.

    Returns:
        tuple[torch.device, DistCoordinator]: The device and the distributed coordinator.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # NOTE: A very large timeout is set to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataset and dataloader
    if training_config["dataset"]["type"] == "word":
        dataset = T3DataSetWarp(
            glyph_scale=training_config["dataset"].get("glyph_scale", 1),
            condition_type=training_config["condition_type"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
            condition_size=training_config["dataset"].get("condition_size", None),
            random_select=training_config["dataset"].get("random_select", False),
        )
    else:
        raise NotImplementedError

    print("Dataset length:", len(dataset))

    pin_memory_cache_pre_alloc_numels = None
    cache_pin_memory = pin_memory_cache_pre_alloc_numels is not None
    num_groups = int(os.environ.get("WORLD_SIZE", 1))
    dataloader_args = dict(
        dataset=dataset,
        batch_size=training_config.get("batch_size", None),
        num_workers=training_config.get("dataloader_workers", 4),
        seed=training_config.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group={'size': num_groups, 'rank': int(os.environ.get("RANK", '0'))},
        prefetch_factor=training_config.get("prefetch_factor", None),
        cache_pin_memory=cache_pin_memory,
        num_groups=num_groups,
    )
    train_loader, sampler = prepare_dataloader(
        bucket_config=training_config.get("bucket_config", None),
        num_bucket_build_workers=training_config.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    print("after prepare_dataloader")
    num_steps_per_epoch = len(train_loader)

    # Initialize model
    if 'model_type' in config:
        if config['model_type'] == 'flux_fill':
            trainable_model = OminiModelFIll(
                flux_pipe_id=config["flux_path"],
                reuse_lora_path=training_config.get("reuse_lora_path", None),
                lora_config=training_config["lora_config"],
                device=f"cuda",
                dtype=getattr(torch, config["dtype"]),
                optimizer_config=training_config["optimizer"],
                model_config=config.get("model", {}),
                gradient_checkpointing=training_config.get("gradient_checkpointing", False),
                odm_loss_config=training_config.get("odm_loss", None),
                ocr_loss_config=training_config.get("ocr_loss", None),
                byt5_encoder_config=training_config.get("byt5_encoder", None)
            )
        else:
            raise NotImplementedError

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    node_num = int(os.environ.get("NODE_NUM", 1))
    print(f"World size: {world_size}")
    print(f"Number of nodes: {node_num}")
    # Initialize trainer
    trainer = L.Trainer(
        num_nodes=node_num,
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        use_distributed_sampler=False
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}", exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    # 初始化加速器
    setup_device()
    main()
