# Refactoring and GRPO-RL Integration Changelog

This document summarizes the major refactoring and feature additions to the training pipeline, culminating in the integration of a full Generative Reward Policy Optimization (GRPO) / Proximal Policy Optimization (PPO) training scheme for IP-Adapter.

## 1. Modular, Multi-Model Architecture (Unchanged)

The core training scripts (`train/model.py` and `train/train.py`) were refactored to support multiple underlying transformer architectures in a plug-and-play manner.

- **`--model_type` Argument**: A command-line argument `--model_type` allows switching between models (e.g., `flux`, `qwen`).
- **Conditional Imports**: Scripts dynamically load the correct model classes, attention processors, and utility functions based on `model_type`.
- **Model-Specific Logic Paths**: The main training loop handles model-specific data preparation (e.g., latent packing for Flux) and post-processing.
- **Directory Structure**: `train/flux_ip/` and `train/qwen_ip/` house model-specific modules.

## 2. Overhauled Data Loading Pipeline (Unchanged)

The data loading mechanism in `train/dataset.py` was rewritten to support a custom local dataset format.

- **Parsing `self_bench.txt`**: The `SimpleDataset` class reads a `self_bench.txt` file to get sample numbers and text prompts.
- **Image Triplet Loading**: The dataset loads triplets for inpainting: `_source.png`, `_mask.png`, and `_ref.png`.
- **Role-Specific Processing**:
  - **Ground Truth (`pixel_values`)**: The `ref` image is the target for the loss function.
  - **Input (`source_image`)**: The `source` image is used as context.
  - **Style Image (`style_images`)**: A crop from the `ref` image is used as input for the IP-Adapter.

## 3. Full GRPO/PPO Reinforcement Learning Integration

The training pipeline was completely re-architected to support a full reinforcement learning loop, drawing inspiration and core logic from `flow_grpo`. This replaces the previous, incomplete RL implementation.

### 3.1. Decoupled Reward Server

- **`train/rl_ip/reward_server.py`**: A standalone FastAPI server was implemented. It loads both the VLM (QwenVL) and OCR (PaddleOCR) models onto a specified GPU and exposes a single `/score` endpoint. This decouples the memory-intensive reward models from the main training process.
- **`run_server.sh`**: A new launch script was created to easily start one or more instances of the reward server on multiple GPUs, with automatic port assignment and logging.

### 3.2. Parallel Reward Client

- **`train/rl_ip/reward.py`**: A new `RewardClient` was implemented. It can connect to multiple reward server instances and uses a thread pool to send reward requests in parallel using a round-robin distribution. This is crucial for preventing the reward calculation from becoming a bottleneck in multi-GPU training.
- **Weighted Reward Combination**: The client automatically combines the `vlm_score` and `ocr_confidence` from the server based on the `--ocr_weight` and `--vlm_weight` arguments, providing a single `combined_score` to the training loop.

### 3.3. Core RL Logic and Training Loop

- **Abandoned Prototypes**: The old, non-functional RL modules (`grpo_trainer.py`, `policy.py`) were deleted.
- **`train.py` Overhaul**: The main training script was fundamentally restructured into a `while` loop that supports two distinct phases:
  - **Supervised Warmup**: For the initial `--rl_warmup_steps`, the script performs standard supervised training on the IP-Adapter to ensure a stable starting policy.
  - **RL Epochs**: After the warmup, the script enters the main RL loop, which alternates between **Sampling (Rollout)** and **Training**.

### 3.4. Sampling (Rollout) Phase

- **`perform_rollout` Function**: A new, dedicated function was implemented to generate trajectories. It runs a full diffusion denoising process for a batch of prompts.
- **`log_prob` Calculation**: In each denoising step, it uses the critical `sde_step_with_logprob` utility (ported from `flow_grpo`) to calculate the log probability of the chosen action (denoising step).
- **Data Collection**: The rollout phase collects all necessary data for training: `latents`, `next_latents`, `log_probs`, and the `rewards` obtained from the `RewardClient`.

### 3.5. Training Phase (PPO/GRPO)

- **Advantage Calculation (GRPO vs. PPO)**:
  - After sampling, the collected rewards are gathered across all processes.
  - If `--rl_per_prompt_stat_tracking` is enabled (the **GRPO** mode), `PerPromptStatTracker` is used to normalize rewards **on a per-prompt basis** before calculating advantages. This is the key feature of GRPO.
  - If disabled, advantages are calculated using standard normalization across the entire batch (standard **PPO**).
- **PPO-Clip Loss Function**: A `compute_log_prob` function was implemented to re-evaluate actions with the current policy. The training loop then calculates the importance sampling ratio (`ratio`) and applies the PPO-Clip surrogate objective function to compute a stable `policy_loss`.
- **Inner Epochs**: The collected data is reused for multiple training iterations (`--rl_num_inner_epochs`), improving data efficiency.


INFO:__main__:mask_latents shape: torch.Size([1, 1, 64,64]) 0/400 [00:00<?, ?it/s]
INFO:__main__:masked_image_latents shape: torch.Size([1, 16, 64, 64])
INFO:__main__:packed_noisy_latents shape: torch.Size([1, 1024, 64])
INFO:__main__:packed_masked_image_latents shape: torch.Size([1, 1024, 64])
INFO:__main__:mask_latents shape: torch.Size([1, 1, 64, 64])
INFO:__main__:masked_image_latents shape: torch.Size([1, 16, 64, 64])
INFO:__main__:packed_noisy_latents shape: torch.Size([1, 1024, 64])
INFO:__main__:packed_masked_image_latents shape: torch.Size([1, 1024, 64])