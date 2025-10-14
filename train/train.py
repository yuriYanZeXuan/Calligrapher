import argparse
import logging
import math
import os
import sys
from pathlib import Path

# Add the project's root directory ('Calligrapher') to the Python path.
# This allows for absolute imports from the project root.
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import itertools
from functools import partial
import torch.nn.functional as F

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from tqdm.auto import tqdm
from transformers import AutoProcessor

from train.dataset import SimpleDataset, collate_fn
from train.model import (
    load_text_encoders_and_tokenizers,
    load_vae_and_transformer,
    load_image_encoder,
    setup_ip_adapter,
)
# Note: utils will be conditionally imported based on model type.
# from utils import (
#     encode_prompt,
#     pack_latents,
#     unpack_latents,
#     prepare_latent_image_ids,
#     get_sigmas,
#     prepare_mask_latents4training,
# )
from train.rl_ip.policy import create_policy
from train.rl_ip.reward import RewardCalculator
from train.rl_ip.grpo_trainer import GRPOTrainer

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple training script for FLUX IP-Adapter.")
    # --- Model and Paths ---
    parser.add_argument("--model_type", type=str, default="flux", choices=["flux", "qwen"], help="Type of model to train.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--siglip_path", type=str, required=True, help="Path to SigLIP model.")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    # --- Dataset ---
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--train_data_json", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--max_train_samples", type=int, default=None)
    
    # --- Training Params ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # --- Text Encoding ---
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum sequence length for text prompt.")

    # --- Diffusion ---
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal")

    # --- RL Training ---
    parser.add_argument("--use_rl", action="store_true", help="Use Reinforcement Learning (GRPO) for training.")
    parser.add_argument("--rl_warmup_steps", type=int, default=1000, help="Number of supervised steps before starting RL training.")
    parser.add_argument("--ocr_weight", type=float, default=0.6, help="Weight for OCR reward in RL.")
    parser.add_argument("--vlm_weight", type=float, default=0.4, help="Weight for VLM score in RL.")
    parser.add_argument("--vlm_model_path", type=str, default="Qwen/Qwen-VL-Chat", help="Path to the VLM model for reward calculation.")


    # --- Logging ---
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")
    
    args = parser.parse_args()
    
    if args.model_type == 'flux':
        global encode_prompt, pack_latents, unpack_latents, prepare_latent_image_ids, get_sigmas, prepare_mask_latents4training
        from train.flux_ip.utils import (
            encode_prompt, pack_latents, unpack_latents, prepare_latent_image_ids, get_sigmas, prepare_mask_latents4training
        )
    elif args.model_type == 'qwen':
        # NOTE: You need to create and implement qwen_ip/utils.py with functions
        # that are compatible with the Qwen model's architecture.
        # For now, we'll try to use flux's utils as placeholders.
        logger.warning("Using placeholder utils from 'flux_ip' for 'qwen' model. These may need to be adapted.")
        global encode_prompt, pack_latents, unpack_latents, prepare_latent_image_ids, get_sigmas, prepare_mask_latents4training
        from train.flux_ip.utils import (
            encode_prompt, pack_latents, unpack_latents, prepare_latent_image_ids, get_sigmas, prepare_mask_latents4training
        )
    else:
        raise ValueError(f"Unknown model_type '{args.model_type}' for utility loading.")

    return args

def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Models ---
    tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two = load_text_encoders_and_tokenizers(args)
    vae, transformer = load_vae_and_transformer(args)
    image_encoder = load_image_encoder(args)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    # Freeze models
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    image_encoder.requires_grad_(False)

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    
    image_proj_model = setup_ip_adapter(transformer, accelerator, weight_dtype, args)

    # --- RL Setup ---
    if args.use_rl:
        policy = create_policy(transformer, noise_scheduler, args.model_type)
        reward_calculator = RewardCalculator(
            accelerator.device, 
            args.ocr_weight, 
            args.vlm_weight, 
            vlm_model_path=args.vlm_model_path
        )
        grpo_trainer = GRPOTrainer(policy, accelerator, lr=args.learning_rate)
    else:
        policy, reward_calculator, grpo_trainer = None, None, None

    # --- Optimizer ---
    attn_processors = transformer.attn_processors.values()
    params_to_optimize = itertools.chain(image_proj_model.parameters(), *(p.parameters() for p in attn_processors))
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)

    # --- Dataset ---
    train_dataset = SimpleDataset(args, accelerator)
    clip_image_processor = AutoProcessor.from_pretrained(args.siglip_path)
    
    # We need to use functools.partial to pass the clip_image_processor to the collate_fn
    collate_fn_with_processor = partial(collate_fn, clip_image_processor=clip_image_processor)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_processor,
        num_workers=args.dataloader_num_workers,
    )
    
    # --- Scheduler and Accelerator ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    transformer, image_proj_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, image_proj_model, optimizer, train_dataloader, lr_scheduler
    )

    # --- Training ---
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    noise_scheduler_copy = noise_scheduler

    for epoch in range(args.num_train_epochs):
        if not args.use_rl or global_step < args.rl_warmup_steps:
            # --- SUPERVISED TRAINING ---
            transformer.train()
            image_proj_model.train()
        else:
            # --- RL TRAINING ---
            policy.train() # The policy contains the transformer
            image_proj_model.train() # The projection model is still trained supervised-style
            
        for step, batch in enumerate(train_dataloader):
            if args.use_rl and global_step >= args.rl_warmup_steps:
                # =======================================================
                # ============== REINFORCEMENT LEARNING STEP ============
                # =======================================================
                with accelerator.accumulate(policy, image_proj_model):
                    # --- 1. Rollout Trajectory ---
                    # This part needs to be carefully implemented to match the diffusion process
                    # and collect all necessary data for the GRPO update.
                    
                    # NOTE: The full rollout implementation is complex and requires careful state management.
                    # This is a conceptual placeholder for the rollout logic.
                    # A real implementation would loop through `num_inference_steps`.
                    
                    # For now, we will perform a simplified one-step GRPO update
                    # as a demonstration, similar to how DPO works.
                    
                    # --- Simplified 1-step GRPO ---
                    
                    # Prepare inputs
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    prompts = batch["prompts"]
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two],
                        prompts, accelerator.device, max_sequence_length=args.max_sequence_length
                    )
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)
                    
                    # Sample a single timestep
                    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                    
                    # Corrupt the latents to s_t
                    noise = torch.randn_like(latents)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, t)

                    # Get model prediction and log_prob for one step
                    # We need a "previous" latent to score, let's use the ground truth `latents`
                    # This is a simplification similar to 1-step DPO-style training.
                    _, log_prob, _, _ = policy(
                        latents=noisy_latents,
                        prev_latents=latents,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        # Pass other required args for the transformer
                        pooled_projections=pooled_prompt_embeds,
                        # ... any other kwargs your model needs
                    )

                    # Generate final image from `noisy_latents` to get reward
                    # For this simplified step, we can't generate a full image.
                    # A full rollout would be needed.
                    # As a placeholder, we'll calculate reward on the ground truth image
                    # and use the log_prob of correcting the noise in one step.
                    rewards = reward_calculator.get_reward(pixel_values, prompts)
                    
                    # GRPO Loss
                    loss = - (rewards * log_prob).mean()
                    
                    # Backprop
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    
                    optimizer.step() # Assuming the PPO optimizer handles the policy params
                    lr_scheduler.step()
                    optimizer.zero_grad()

            else:
                # =======================================================
                # ================ SUPERVISED TRAINING STEP ===============
                # =======================================================
                with accelerator.accumulate(transformer, image_proj_model):
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    prompts = batch["prompts"]
                    
                    # Get text embeddings
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two], 
                        prompts, accelerator.device, max_sequence_length=args.max_sequence_length
                    )

                    # VAE encode
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)

                    # Sample noise and timesteps
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme, 
                        batch_size=bsz,
                        logit_mean=0.0,
                        logit_std=1.0,
                        mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)

                    # Add noise
                    sigmas = get_sigmas(noise_scheduler_copy, timesteps, n_dim=latents.ndim, dtype=latents.dtype, device=latents.device)
                    noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
                    
                    # Prepare mask and masked image latents
                    mask_image = batch["mask"].to(dtype=vae.dtype)
                    source_image = batch["source_image"].to(dtype=vae.dtype)
                    masked_image = source_image * (1 - mask_image)
                    
                    # --- MODEL-SPECIFIC LOGIC ---
                    if args.model_type == 'flux':
                        b, c, h, w = noisy_latents.shape
                        
                        # Prepare mask and masked image latents, which are essential for the inpainting task.
                        mask_latents, masked_image_latents = prepare_mask_latents4training(
                            mask=mask_image,
                            masked_image=masked_image,
                            batch_size=bsz,
                            height=args.resolution,
                            width=args.resolution,
                            dtype=prompt_embeds.dtype,
                            device=accelerator.device,
                            vae=vae,
                            vae_scale_factor=8,
                        )
                        
                        # Concatenate all inpainting-related latents along the channel dimension.
                        # This creates the 33-channel input that the reconfigured model now expects.
                        inpainting_latents = torch.cat((noisy_latents, masked_image_latents, mask_latents), dim=1)
                        
                        # Pack the concatenated latents to create the final hidden states for the transformer.
                        c_inpainting = inpainting_latents.shape[1]
                        transformer_hidden_states = pack_latents(inpainting_latents, b, c_inpainting, h, w)

                    elif args.model_type == 'qwen':
                        # NOTE: This is a placeholder for Qwen's data preparation, mirroring the flux logic.
                        logger.info("Using placeholder data preparation for Qwen model.", main_process_only=True)
                        b, c, h, w = noisy_latents.shape
                        mask_latents, masked_image_latents = prepare_mask_latents4training(
                            mask=mask_image,
                            masked_image=masked_image,
                            batch_size=bsz,
                            height=args.resolution,
                            width=args.resolution,
                            dtype=prompt_embeds.dtype,
                            device=accelerator.device,
                            vae=vae,
                            vae_scale_factor=8,
                        )
                        
                        inpainting_latents = torch.cat((noisy_latents, masked_image_latents, mask_latents), dim=1)
                        
                        c_inpainting = inpainting_latents.shape[1]
                        transformer_hidden_states = pack_latents(inpainting_latents, b, c_inpainting, h, w)


                    # Get image embeddings
                    with torch.no_grad():
                        image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).pooler_output
                    
                    image_embeds_ = []
                    for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            image_embeds_.append(torch.zeros_like(image_embed))
                        else:
                            image_embeds_.append(image_embed)
                    image_embeds = torch.stack(image_embeds_)
                    
                    ip_tokens = image_proj_model(image_embeds)

                    # Model prediction
                    guidance = torch.tensor([1.0], device=accelerator.device).expand(bsz)
                    height, width = latents.shape[2], latents.shape[3]
                    img_ids = prepare_latent_image_ids(height, width, accelerator.device, weight_dtype)
                    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=accelerator.device, dtype=weight_dtype)

                    model_pred = transformer(
                        hidden_states=transformer_hidden_states,
                        timestep=timesteps / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        image_emb=ip_tokens,
                        txt_ids=text_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )[0]
                    
                    # --- MODEL-SPECIFIC POST-PROCESSING ---
                    if args.model_type == 'flux':
                        model_pred = unpack_latents(model_pred, height * 8, width * 8, 16)
                    elif args.model_type == 'qwen':
                        logger.info("Using placeholder post-processing for Qwen model.", main_process_only=True)
                        model_pred = unpack_latents(model_pred, height * 8, width * 8, 16)

                    model_pred = model_pred * (-sigmas) + noisy_latents

                    # Loss calculation
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                    loss = torch.mean((weighting * (model_pred - latents) ** 2).reshape(bsz, -1), 1).mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        attn_processors_unwrapped = accelerator.unwrap_model(transformer).attn_processors.values()
                        params_to_clip = itertools.chain(
                            accelerator.unwrap_model(image_proj_model).parameters(),
                            *(p.parameters() for p in attn_processors_unwrapped)
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()
