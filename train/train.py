import argparse
import logging
import math
import os
import sys
from pathlib import Path
import time
import random
import tempfile
from collections import defaultdict
from PIL import Image
import numpy as np

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
from accelerate.utils import set_seed, gather_object
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
# --- New RL Imports ---
from train.rl_ip.reward import RewardClient
from train.rl_ip.stat_tracking import PerPromptStatTracker
from train.rl_ip.grpo_utils import sde_step_with_logprob


logger = get_logger(__name__)

@torch.no_grad()
def perform_rollout(
    args, vae, transformer, image_proj_model, text_encoders, tokenizers, image_encoder,
    noise_scheduler, batch, weight_dtype, device
):
    """
    Performs a single rollout to generate trajectories (images, latents, log_probs).
    """
    num_images_per_prompt = args.rl_num_images_per_prompt
    
    # 1. Prepare Inputs
    prompts = batch["prompts"]
    
    # --- Offload logic for text_encoder_two ---
    text_encoder_one, text_encoder_two = text_encoders
    text_encoder_two.to(device, dtype=weight_dtype)
    
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        [text_encoder_one, text_encoder_two], tokenizers, prompts, device, max_sequence_length=args.max_sequence_length
    )
    
    text_encoder_two.to("cpu")
    torch.cuda.empty_cache()
    # --- End Offload logic ---
    
    with torch.no_grad():
        image_embeds = image_encoder(batch["clip_images"].to(device, dtype=weight_dtype)).pooler_output
    
    image_embeds_ = []
    for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
        if drop_image_embed == 1:
            image_embeds_.append(torch.zeros_like(image_embed))
        else:
            image_embeds_.append(image_embed)
    image_embeds = torch.stack(image_embeds_)
    
    ip_tokens = image_proj_model(image_embeds)
    
    # --- RL Inpainting Adaptation ---
    # The rollout needs to mimic the inpainting data format that the model expects.
    # We'll use the source image and mask from the batch to create the initial state.
    mask_image = batch["mask"].to(device, dtype=weight_dtype)
    source_image = batch["source_image"].to(device, dtype=weight_dtype)
    
    # VAE encode the source image to get latents that will be combined with noise.
    source_latents = vae.encode(source_image * (1 - mask_image)).latent_dist.sample() * vae.config.scaling_factor
    source_latents = source_latents.to(dtype=weight_dtype)

    # --- Repeat inputs for multiple samples per prompt ---
    prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
    ip_tokens = ip_tokens.repeat(num_images_per_prompt, 1, 1)
    source_latents = source_latents.repeat(num_images_per_prompt, 1, 1, 1)

    # NOTE: We DO NOT repeat mask_image and source_image here.
    # They will be handled inside the loop to avoid double-repeating.
    
    # 2. Prepare Latents
    # Generate a unique noise for each sample
    noise = torch.randn_like(source_latents)
    # Start the diffusion process from the noised source image latents
    # Manually add noise for the first timestep, as FlowMatchEulerDiscreteScheduler lacks a .add_noise() method.
    # We replicate the logic from the supervised training part.
    t_start = noise_scheduler.timesteps[0].repeat(source_latents.shape[0])
    sigmas_start = get_sigmas(noise_scheduler, t_start, n_dim=source_latents.ndim, dtype=source_latents.dtype, device=source_latents.device)
    latents = sigmas_start * noise + (1.0 - sigmas_start) * source_latents

    # --- End RL Inpainting Adaptation ---
    
    # 3. Denoising Loop (Trajectory Generation)
    all_latents = [latents]
    all_log_probs = []
    
    scheduler_timesteps = noise_scheduler.timesteps
    
    for i, t in tqdm(enumerate(scheduler_timesteps), total=len(scheduler_timesteps), desc="Rollout step", leave=False):
        
        # Prepare for model input. Keep t_batch as float.
        t_batch = t.repeat(latents.shape[0]).to(device)
        
        # --- RL Inpainting Adaptation ---
        # Create the packed hidden states required by the inpainting transformer
        b, c, h, w = latents.shape
        
        # Create the masked image for the VAE. This needs to be pre-repeated to match the batch size `b`.
        source_image_repeated = batch["source_image"].to(device, dtype=weight_dtype).repeat(b, 1, 1, 1)
        mask_repeated_for_mult = batch["mask"].to(device, dtype=weight_dtype).repeat(b, 1, 1, 1)
        masked_image_for_prep = source_image_repeated * (1 - mask_repeated_for_mult)
        
        # Call the utils function with the correct inputs:
        # - mask: The original, un-repeated mask (the util handles its batching).
        # - masked_image: The pre-repeated image (the util does not handle its batching).
        mask_latents, masked_image_latents = prepare_mask_latents4training(
            mask=batch["mask"].to(device, dtype=weight_dtype),
            masked_image=masked_image_for_prep,
            batch_size=b,
            height=args.resolution,
            width=args.resolution,
            dtype=prompt_embeds.dtype,
            device=device,
            vae=vae,
            vae_scale_factor=8,
        )

        packed_noisy_latents = pack_latents(latents, b, c, h, w)
        packed_masked_image_latents = pack_latents(masked_image_latents, b, c, h, w)
        
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        mask_c = vae_scale_factor ** 2
        packed_mask = pack_latents(mask_latents.repeat(1, mask_c, 1, 1), b, mask_c, h, w)

        transformer_hidden_states = torch.cat(
            [packed_noisy_latents, packed_masked_image_latents, packed_mask], dim=-1
        )
        # --- End RL Inpainting Adaptation ---

        # Model prediction
        guidance = torch.tensor([args.rl_guidance_scale], device=device).expand(latents.shape[0])
        height, width = latents.shape[2], latents.shape[3]
        img_ids = prepare_latent_image_ids(height, width, device, weight_dtype)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=weight_dtype)
        
        # This part needs to be adapted if the RL task is inpainting.
        # For now, assuming a text-to-image task for RL.
        # b, c, h, w = latents.shape
        # dummy_packed_latents = torch.zeros(b, h*w, 384, device=device, dtype=weight_dtype)

        model_pred = transformer(
            hidden_states=transformer_hidden_states, # This needs to be adapted for T2I
            timestep=t_batch / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            image_emb=ip_tokens,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        
        model_pred = unpack_latents(model_pred, height * 8, width * 8, 16)
        
        # Get sigmas directly using the loop index `i` to avoid floating point comparison issues in `get_sigmas`.
        sigma_t = noise_scheduler.sigmas[i]
        sigmas = sigma_t.repeat(latents.shape[0]).to(dtype=latents.dtype, device=latents.device)
        sigmas = sigmas.view(-1, *([1] * (latents.ndim - 1)))
        
        pred_x0 = model_pred * (-sigmas) + latents
        
        # SDE step to get next latents and log prob
        latents, log_prob, _, _ = sde_step_with_logprob(
            scheduler=noise_scheduler,
            model_output=pred_x0,
            timestep=t_batch,
            sample=latents,
        )
        
        all_latents.append(latents)
        all_log_probs.append(log_prob)

    # 4. Decode Final Latent to Image
    latents = latents / vae.config.scaling_factor
    images_tensor = vae.decode(latents.to(vae.dtype)).sample

    # --- Sanitize and Normalize Output ---
    # Replace any NaNs or Infs that can occur during RL training with a stable value.
    images_tensor = torch.nan_to_num(images_tensor)
    # Clamp the values to the expected [-1, 1] range, then normalize to [0, 1]
    images_tensor = (images_tensor / 2 + 0.5).clamp(0, 1)
    
    # Convert to PIL images
    images_pil = [(t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for t in images_tensor]
    images_pil = [Image.fromarray(img) for img in images_pil]

    return images_pil, all_latents, all_log_probs

def compute_log_prob(
    args, vae, transformer, image_proj_model, text_encoders, tokenizers, image_encoder,
    noise_scheduler, sample, timestep_idx, weight_dtype, device
):
    """
    Computes the log probability of a given transition using the current model.
    """
    # 1. Prepare Inputs from the sampled data
    prompts = sample["prompts"]
    
    # --- Offload logic for text_encoder_two ---
    text_encoder_one, text_encoder_two = text_encoders
    text_encoder_two.to(device, dtype=weight_dtype)

    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        [text_encoder_one, text_encoder_two], tokenizers, prompts, device, max_sequence_length=args.max_sequence_length
    )

    text_encoder_two.to("cpu")
    torch.cuda.empty_cache()
    # --- End Offload logic ---
    
    with torch.no_grad():
        image_embeds = image_encoder(sample["clip_images"].to(device, dtype=weight_dtype)).pooler_output
        
    image_embeds_ = []
    for image_embed, drop_image_embed in zip(image_embeds, sample["drop_image_embeds"]):
        if drop_image_embed == 1:
            image_embeds_.append(torch.zeros_like(image_embed))
        else:
            image_embeds_.append(image_embed)
    image_embeds = torch.stack(image_embeds_)
    
    ip_tokens = image_proj_model(image_embeds)
    
    # 2. Get latents and timestep for the specific step
    latents = sample["latents"][:, timestep_idx]
    next_latents = sample["next_latents"][:, timestep_idx]
    t = noise_scheduler.timesteps[timestep_idx]
    # Keep t_batch as float.
    t_batch = t.repeat(latents.shape[0]).to(device)

    # 3. Model Prediction (similar to rollout but with gradients)
    guidance = torch.tensor([args.rl_guidance_scale], device=device).expand(latents.shape[0])
    height, width = latents.shape[2], latents.shape[3]
    img_ids = prepare_latent_image_ids(height, width, device, weight_dtype)
    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=weight_dtype)
    
    # This is a simplification, we should really be carrying the clip images with the prompts
    # Re-create the mask and source from the batch
    # We assume the `sample` dict now contains the necessary info.
    # This requires modification upstream where `all_samples` is created.
    # For now, let's assume we can get it. This will likely fail if not fixed.
    
    # FIXME: The source_image and mask need to be passed along with the sample.
    # For now, we will have to use a placeholder or data from the last batch from the parent scope,
    # which is not ideal.
    # A proper fix involves adding "source_image" and "mask" to the `all_samples` dict.
    
    b, c, h, w = latents.shape
    # NOTE: Using sample['source_image'] and sample['mask'] which we assume are now passed.
    # We need to repeat them here as well to match the batch size of latents,
    # as the latents come from the full trajectory of all generated images.
    num_images_per_prompt = latents.shape[0] // sample["source_image"].shape[0]
    
    source_image_repeated = sample["source_image"].to(device, dtype=weight_dtype).repeat(num_images_per_prompt, 1, 1, 1)
    mask_repeated_for_mult = sample["mask"].to(device, dtype=weight_dtype).repeat(num_images_per_prompt, 1, 1, 1)
    masked_image_for_prep = source_image_repeated * (1 - mask_repeated_for_mult)

    mask_latents, masked_image_latents = prepare_mask_latents4training(
        mask=sample["mask"].to(device, dtype=weight_dtype), # Pass original mask
        masked_image=masked_image_for_prep, # Pass pre-repeated masked image
        batch_size=b,
        height=args.resolution,
        width=args.resolution,
        dtype=prompt_embeds.dtype,
        device=device,
        vae=vae,
        vae_scale_factor=8,
    )

    packed_noisy_latents = pack_latents(latents, b, c, h, w)
    packed_masked_image_latents = pack_latents(masked_image_latents, b, c, h, w)
    
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    mask_c = vae_scale_factor ** 2
    packed_mask = pack_latents(mask_latents.repeat(1, mask_c, 1, 1), b, mask_c, h, w)

    transformer_hidden_states = torch.cat(
        [packed_noisy_latents, packed_masked_image_latents, packed_mask], dim=-1
    )


    model_pred = transformer(
        hidden_states=transformer_hidden_states,
        timestep=t_batch / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        image_emb=ip_tokens,
        txt_ids=text_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]

    model_pred = unpack_latents(model_pred, height * 8, width * 8, 16)
    sigmas = get_sigmas(noise_scheduler, t_batch, n_dim=latents.ndim, dtype=latents.dtype, device=latents.device)
    pred_x0 = model_pred * (-sigmas) + latents

    # 4. Compute log prob for the transition to `next_latents`
    _, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        scheduler=noise_scheduler,
        model_output=pred_x0,
        timestep=t_batch,
        sample=latents,
        prev_sample=next_latents, # Provide the actual next latent from trajectory
    )
    
    return log_prob, prev_sample_mean, std_dev_t


def log_memory_breakdown(accelerator, title, models_dict, params_to_optimize=None):
    if not accelerator.is_main_process:
        return

    torch.cuda.synchronize()
    
    total_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    total_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

    print(f"\n--- Memory Breakdown: {title} ---")
    print(f"Total GPU Memory Allocated: {total_allocated_gb:.2f} GB | Reserved: {total_reserved_gb:.2f} GB")

    def get_model_size_gb(model):
        return sum(p.numel() * p.element_size() for p in model.parameters() if p.device.type == 'cuda') / (1024 ** 3)

    # --- 1. Model Weights ---
    print(f"\n[1. Model Weights]")
    total_model_size = 0
    for name, model in models_dict.items():
        size = get_model_size_gb(model)
        print(f"  - {name}: {size:.3f} GB")
        total_model_size += size
    print(f"  ---------------------------------")
    print(f"  Total Model Weights: {total_model_size:.3f} GB")

    # --- 2. Gradients ---
    gradient_size = 0
    if params_to_optimize:
        gradient_size = sum(p.grad.numel() * p.grad.element_size() for p in params_to_optimize if p.grad is not None and p.grad.device.type == 'cuda') / (1024 ** 3)
    
    if gradient_size > 0:
        print(f"\n[2. Gradients (for trainable params)]")
        print(f"  - Gradient Size: {gradient_size:.3f} GB")

    # --- 3. Optimizer States, Activations & Other ---
    accounted_for = total_model_size + gradient_size
    other_mem = total_allocated_gb - accounted_for
    
    print(f"\n[3. Other (Optimizer States, Activations, Buffers)]")
    print(f"  - Estimated Size: {max(0, other_mem):.3f} GB")
    print("--------------------------------------------------\n")


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
    parser.add_argument("--vlm_model_path", type=str, default="Qwen/Qwen-VL-Chat", help="Path to the VLM model for reward calculation (used if not using API).")
    parser.add_argument("--reward_server_url", type=str, default="http://127.0.0.1:8000/score", help="URL of the reward model API server.")

    # --- Memory/Speed Optimizations ---
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--no_rl_reward_model", action="store_true", help="Disable loading the VLM reward model and use random rewards for testing.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit AdamW optimizer.")
    parser.add_argument("--enable_memory_profiler", action="store_true", help="Enable detailed memory usage logging.")

    # --- Logging ---
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")

    # --- RL Sampling and Training ---
    parser.add_argument("--rl_num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt during RL.")
    parser.add_argument("--rl_per_prompt_stat_tracking", action="store_true", help="Enable per-prompt advantage normalization (GRPO).")
    parser.add_argument("--rl_num_batches_per_epoch", type=int, default=1, help="Number of batches to sample per RL epoch.")
    parser.add_argument("--rl_num_inference_steps", type=int, default=50, help="Number of diffusion steps for sampling.")
    parser.add_argument("--rl_guidance_scale", type=float, default=3.5, help="Guidance scale for sampling.")
    parser.add_argument("--rl_timestep_fraction", type=float, default=1.0, help="Fraction of timesteps to train on per trajectory.")
    parser.add_argument("--rl_num_inner_epochs", type=int, default=1, help="Number of training epochs on the sampled data.")
    parser.add_argument("--rl_adv_clip_max", type=float, default=5, help="Max value for advantage clipping.")
    parser.add_argument("--rl_grpo_clip_range", type=float, default=0.2, help="PPO clipping range for GRPO.")
    parser.add_argument("--rl_kl_beta", type=float, default=0.1, help="Beta coefficient for the KL penalty term.")
    
    
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

    # Force resolution to 512x512 if it's higher, to save memory.
    if args.resolution > 512:
        logger.warning(f"Resolution {args.resolution} is too high. Forcing to 512x512 to save memory.")
        args.resolution = 512

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
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    
    # Offload text_encoder_two to CPU to save VRAM. It will be moved to the GPU only when needed.
    if accelerator.is_main_process:
        logger.info("Offloading text_encoder_two to CPU to save VRAM.")
    text_encoder_two.to("cpu")
    
    if args.gradient_checkpointing:
        transformer.gradient_checkpointing = True
        text_encoder_one.gradient_checkpointing_enable()
        text_encoder_two.gradient_checkpointing_enable()
        image_encoder.gradient_checkpointing_enable()

    image_proj_model = setup_ip_adapter(transformer, accelerator, weight_dtype, args)

    if args.enable_memory_profiler:
        models_dict = {
            "VAE": vae,
            "Text Encoder 1": text_encoder_one,
            "Text Encoder 2": text_encoder_two,
            "Image Encoder": image_encoder,
            "Transformer": accelerator.unwrap_model(transformer),
            "IP Proj Model": image_proj_model,
        }
        log_memory_breakdown(accelerator, "After Model Loading", models_dict)

    # --- RL Setup ---
    if args.use_rl:
        if not args.no_rl_reward_model:
            # FIX: The shell script now provides a comma-separated list. Simply split it.
            reward_server_urls = args.reward_server_url.split(',')
            logger.info(f"Process {accelerator.process_index} connecting to reward servers: {reward_server_urls}")
            reward_client = RewardClient(
                server_urls=reward_server_urls, 
                ocr_weight=args.ocr_weight, 
                vlm_weight=args.vlm_weight
            )
        else:
            # This is a dummy client if rewards are disabled.
            class DummyRewardClient:
                def get_rewards_batch(self, images, prompts):
                    logger.warning("Using dummy reward client. Generating random rewards.")
                    return [{'combined_score': random.random()} for _ in images]
            reward_client = DummyRewardClient()

        if args.rl_per_prompt_stat_tracking:
            stat_tracker = PerPromptStatTracker()

    # --- Optimizer ---
    if args.use_8bit_adam:
        try:
            import bitsandbytes.optim as bnb_optim
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        logger.info("Using 8-bit AdamW optimizer.")
        optimizer_cls = bnb_optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    attn_processors = transformer.attn_processors.values()
    params_to_optimize = itertools.chain(image_proj_model.parameters(), *(p.parameters() for p in attn_processors))
    optimizer = optimizer_cls(params_to_optimize, lr=args.learning_rate)

    # --- Dataset ---
    train_dataset = SimpleDataset(args, accelerator)
    clip_image_processor = AutoProcessor.from_pretrained(args.siglip_path, use_fast=True)
    
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
    
    # The dataloader iterator for getting prompts
    train_iter = iter(train_dataloader)

    # RL training loop
    while global_step < args.max_train_steps:
        
        if not args.use_rl or global_step < args.rl_warmup_steps:
            # =======================================================
            # ================ SUPERVISED TRAINING STEP ===============
            # =======================================================
            transformer.train()
            image_proj_model.train()
            
            # This will be filled in the next step, for now just the structure.
            with accelerator.accumulate(transformer, image_proj_model):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)

                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                prompts = batch["prompts"]
                
                # Get text embeddings
                # --- Offload logic for text_encoder_two ---
                text_encoder_two.to(accelerator.device, dtype=weight_dtype)
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two], 
                    prompts, accelerator.device, max_sequence_length=args.max_sequence_length
                )
                text_encoder_two.to("cpu")
                torch.cuda.empty_cache()
                # --- End Offload logic ---

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
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                # Add noise
                sigmas = get_sigmas(noise_scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype, device=latents.device)
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
                    
                    # 1. Pack noisy_latents, masked_image_latents, and mask separately, mimicking the official flux-fill pipeline.
                    packed_noisy_latents = pack_latents(noisy_latents, b, c, h, w) # (B, L, 64)
                    
                    packed_masked_image_latents = pack_latents(masked_image_latents, b, c, h, w) # (B, L, 64)
                    
                    # The mask is 1 channel, but needs to be packed. The official pipeline expands it to vae_scale_factor^2 channels.
                    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                    mask_c = vae_scale_factor ** 2 # Should be 8*8 = 64
                    packed_mask = pack_latents(mask_latents.repeat(1, mask_c, 1, 1), b, mask_c, h, w) # (B, L, 256)

                    # 2. Concatenate the packed latents along the feature dimension to create the final 384-dim input.
                    transformer_hidden_states = torch.cat(
                        [packed_noisy_latents, packed_masked_image_latents, packed_mask], dim=-1
                    )

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
                    
                    packed_noisy_latents = pack_latents(noisy_latents, b, c, h, w)
                    packed_masked_image_latents = pack_latents(masked_image_latents, b, c, h, w)
                    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                    mask_c = vae_scale_factor ** 2
                    packed_mask = pack_latents(mask_latents.repeat(1, mask_c, 1, 1), b, mask_c, h, w)

                    transformer_hidden_states = torch.cat(
                        [packed_noisy_latents, packed_masked_image_latents, packed_mask], dim=-1
                    )


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
                
                if args.enable_memory_profiler and accelerator.sync_gradients:
                    log_memory_breakdown(accelerator, f"Step {global_step} - Before Optimizer", models_dict, params_to_optimize)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            
        else:
            # =======================================================
            # ============== REINFORCEMENT LEARNING EPOCH =============
            # =======================================================
            
            # ----------------- SAMPLING (ROLLOUT) -----------------
            logger.info(f"RL Epoch: Starting Rollout Phase for {args.rl_num_batches_per_epoch} batches...")
            
            all_samples = []
            for i in range(args.rl_num_batches_per_epoch):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)
                
                # Generate trajectories
                images_pil, all_latents, all_log_probs = perform_rollout(
                    args, vae, transformer, image_proj_model, 
                    [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two],
                    image_encoder, noise_scheduler, batch, weight_dtype, accelerator.device
                )

                # Get rewards
                prompts = batch["prompts"]
                # Repeat prompts for each generated image to match the reward client's input format
                prompts_for_reward = [p for p in prompts for _ in range(args.rl_num_images_per_prompt)]
                rewards_data = reward_client.get_rewards_batch(images_pil, prompts_for_reward)
                
                # Collate data
                latents_tensor = torch.stack(all_latents, dim=1) # (B * N, T+1, C, H, W)
                log_probs_tensor = torch.stack(all_log_probs, dim=1) # (B * N, T)
                rewards_tensor = torch.tensor([r['combined_score'] for r in rewards_data], device=accelerator.device, dtype=torch.float32)
                
                # --- RL Inpainting Adaptation ---
                # Pass source and mask for re-computation of hidden states during training
                source_image_unrepeated = batch["source_image"]
                mask_unrepeated = batch["mask"]

                num_repeats = args.rl_num_images_per_prompt

                all_samples.append({
                    "prompts": prompts_for_reward,
                    "latents": latents_tensor[:, :-1], # x_t
                    "next_latents": latents_tensor[:, 1:],  # x_{t-1}
                    "log_probs": log_probs_tensor,
                    "rewards": rewards_tensor,
                    # --- RL Inpainting Adaptation ---
                    "source_image": source_image_unrepeated.repeat(num_repeats, 1, 1, 1),
                    "mask": mask_unrepeated.repeat(num_repeats, 1, 1, 1),
                    "clip_images": batch["clip_images"].repeat(num_repeats, 1, 1, 1),
                    "drop_image_embeds": batch["drop_image_embeds"].repeat(num_repeats),
                    # --- End RL Inpainting Adaptation ---
                })
            
            # Collate all batches into a single dictionary of tensors
            samples = {
                k: (torch.cat([s[k] for s in all_samples]) if torch.is_tensor(all_samples[0][k]) else [p for s in all_samples for p in s[k]])
                for k in all_samples[0].keys()
            }
            
            # ---Explicitly free memory from the rollout phase before starting training---
            del all_samples
            torch.cuda.empty_cache()

            logger.info("RL Epoch: Rollout Phase Finished.")
            
            # ----------------- COMPUTE ADVANTAGES -----------------
            logger.info("RL Epoch: Computing Advantages...")
            
            # Gather rewards and prompts across all processes
            gathered_rewards = accelerator.gather(samples["rewards"]).cpu().numpy()
            
            # --- Gather prompts from all processes using the modern `gather_object` utility ---
            prompts_this_process = samples["prompts"]
            all_gathered_prompts = gather_object(prompts_this_process)

            if accelerator.is_main_process:
                if args.rl_per_prompt_stat_tracking:
                    advantages = stat_tracker.update(all_gathered_prompts, gathered_rewards)
                else:
                    advantages = (gathered_rewards - gathered_rewards.mean()) / (gathered_rewards.std() + 1e-8)
            
                # Broadcast advantages to all processes
                advantages_tensor = torch.from_numpy(advantages).to(accelerator.device)
            else:
                advantages_tensor = torch.empty(len(gathered_rewards), device=accelerator.device)
                
            accelerator.wait_for_everyone()
            torch.distributed.broadcast(advantages_tensor, src=0)
            
            # Ungather advantages to get the ones for this process
            batch_size_per_process = len(samples["rewards"])
            start_idx = accelerator.process_index * batch_size_per_process
            end_idx = start_idx + batch_size_per_process
            samples["advantages"] = advantages_tensor[start_idx:end_idx]
            
            # ----------------- TRAINING -----------------
            logger.info("RL Epoch: Starting Training Phase...")
            
            num_train_timesteps = int(args.rl_num_inference_steps * args.rl_timestep_fraction)
            
            for _ in range(args.rl_num_inner_epochs):
                # Shuffle samples
                perm = torch.randperm(len(samples["prompts"]), device=accelerator.device)
                
                # Create a list of indices for non-tensor data
                perm_list = perm.cpu().tolist()
                
                shuffled_samples = {
                    k: (v[perm] if torch.is_tensor(v) else ([v[i] for i in perm_list] if isinstance(v, list) else v))
                    for k, v in samples.items()
                }

                
                for i in tqdm(range(0, len(shuffled_samples["prompts"]), args.train_batch_size), desc="RL Inner Epoch", leave=False):
                    # We need to get the original un-repeated source/mask/clip images for this mini_batch
                    # The `prompts` in the mini_batch are already expanded.
                    # We can find the original items by looking at the dataloader's batch.
                    # This is complex. A simpler way is to carry the un-repeated items along.
                    # The current implementation has a flaw here. Let's fix where data is stored.
                    
                    # The `shuffled_samples` contains repeated source_image and mask.
                    # We need to handle this correctly. Let's adjust the data collation.
                    mini_batch = {k: (v[i:i+args.train_batch_size] if isinstance(v, (torch.Tensor, list)) else v) for k, v in shuffled_samples.items()}

                    for j in tqdm(range(num_train_timesteps), desc="Timestep", leave=False):
                        with accelerator.accumulate(transformer, image_proj_model):
                            log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                                args, vae, transformer, image_proj_model,
                                [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two],
                                image_encoder, noise_scheduler, mini_batch, j, weight_dtype, accelerator.device
                            )
                            
                            # GRPO/PPO loss calculation
                            ratio = torch.exp(log_prob - mini_batch["log_probs"][:, j])
                            advantages_batch = mini_batch["advantages"]
                            
                            unclipped_loss = -advantages_batch * ratio
                            clipped_loss = -advantages_batch * torch.clamp(
                                ratio, 1.0 - args.rl_grpo_clip_range, 1.0 + args.rl_grpo_clip_range
                            )
                            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            
                            loss = policy_loss
                            
                            # Optional KL penalty
                            if args.rl_kl_beta > 0:
                                # To compute KL, we need a reference model. We can use the original frozen transformer.
                                # For simplicity, let's skip this for now, but this is where it would go.
                                # with torch.no_grad():
                                #    _, prev_sample_mean_ref, _ = compute_log_prob(...) using original model
                                # kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean() / (2 * std_dev_t ** 2)
                                # loss += args.rl_kl_beta * kl_loss
                                pass

                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                params_to_clip = itertools.chain(
                                    accelerator.unwrap_model(image_proj_model).parameters(),
                                    *(p.parameters() for p in accelerator.unwrap_model(transformer).attn_processors.values())
                                )
                                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                            
                                if args.enable_memory_profiler:
                                    log_memory_breakdown(accelerator, f"Step {global_step} - Before Optimizer", models_dict, params_to_optimize)

                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()

                        # Checks if the accelerator has performed an optimization step behind the scenes
                        if accelerator.sync_gradients:
                            progress_bar.update(1)
                            global_step += 1
                            
                            # Logging
                            log_data = {
                                "policy_loss": policy_loss.item(),
                                "ratio_mean": ratio.mean().item(),
                                "adv_mean": advantages_batch.mean().item(),
                            }
                            accelerator.log(log_data, step=global_step)
                            progress_bar.set_postfix(**log_data)
                            
                            if global_step >= args.max_train_steps:
                                break
                    if global_step >= args.max_train_steps:
                        break
                if global_step >= args.max_train_steps:
                    break
            
            # Checkpointing logic should be outside the inner loops but inside the main while loop
            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

    accelerator.wait_for_everyone()
    accelerator.end_training()

def cli_main():
    # The main logic is now synchronous.
    main()

if __name__ == "__main__":
    cli_main()
