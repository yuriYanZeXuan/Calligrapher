# Refactoring and GRPO-RL Integration Changelog

This document summarizes the major refactoring and feature additions to the training pipeline, culminating in the integration of a Generative Reward Policy Optimization (GRPO) training scheme for IP-Adapter.

## 1. Modular, Multi-Model Architecture

The core training scripts (`train/model.py` and `train/train.py`) were significantly refactored to support multiple underlying transformer architectures in a plug-and-play manner.

- **`--model_type` Argument**: A new command-line argument `--model_type` was introduced in `train.py`, allowing the user to switch between different models (e.g., `flux`, `qwen`).

- **Conditional Imports**: `model.py` and `train.py` now use conditional imports to dynamically load the correct model classes, attention processors, and utility functions based on the selected `model_type`.

- **Model-Specific Logic Paths**: The main training loop in `train.py` now contains `if/elif` blocks to handle model-specific data preparation (e.g., latent packing for Flux) and post-processing, ensuring that each architecture's unique requirements are met.

- **New Directory Structure**: To support this, new directories were created:
  - `train/flux_ip/`: Contains modules specific to the Flux model.
  - `train/qwen_ip/`: Contains placeholder modules for the Qwen-Edit model.

## 2. Overhauled Data Loading Pipeline

The data loading mechanism in `train/dataset.py` was completely rewritten to support a custom local dataset format instead of relying on Hugging Face `datasets`.

- **Parsing `self_bench.txt`**: The `SimpleDataset` class now reads a `self_bench.txt` file from the training directory to get the sample number and corresponding text prompt for each entry.

- **Image Triplet Loading**: For each sample, the dataset now loads a triplet of images:
  - `testxx_source.png`: The source image with an area to be inpainted.
  - `testxx_mask.png`: The mask defining the inpainting region.
  - `testxx_ref.png`: The reference image providing the target content/style.

- **Role-Specific Processing**: The loaded images are assigned specific roles for the inpainting task:
  - **Ground Truth (`pixel_values`)**: The `ref` image is now the target for the loss function.
  - **Input (`source_image`)**: The `source` image is used as the context for inpainting.
  - **Style Image (`style_images`)**: A crop from the `ref` image (based on the mask's bounding box) is used as the input for the IP-Adapter.

## 3. GRPO Reinforcement Learning Integration

A complete Reinforcement Learning pipeline based on the Generative Reward Policy Optimization (GRPO) algorithm was implemented to allow for direct optimization of the model based on custom rewards.

- **New `rl_ip` Directory**: A dedicated `train/rl_ip/` directory was created to house all RL-related modules.

- **Reward Models**:
  - `rl_ip/ocr.py`: Implements an `OCRScorer` using `PaddleOCR` to evaluate the textual accuracy of generated images.
  - `rl_ip/qwenvl.py`: Implements a `QwenVLScorer` to evaluate the semantic alignment between the generated image and the prompt using the Qwen-VL model.
  - `rl_ip/reward.py`: Contains a `RewardCalculator` that combines scores from the OCR and VLM models into a single, configurable reward signal.

- **Policy and `log_prob` Calculation**:
  - `rl_ip/grpo_utils.py`: Implements the core `sde_step_with_logprob` function, which calculates the log probability of a single denoising step in the diffusion process. This is the mathematical foundation of applying policy gradients to diffusion models.
  - `rl_ip/policy.py`: The `PolicyWrapper` class was updated to use `sde_step_with_logprob`, turning the transformer model into a stochastic policy that can be optimized with RL.

- **GRPO Trainer**:
  - `rl_ip/grpo_trainer.py`: Implements a `GRPOTrainer` responsible for calculating the GRPO loss (`-reward * sum(log_probs)`) and performing backpropagation on the policy network.

- **Integration into `train.py`**:
  - **`--use_rl` Flag**: The main training script now accepts a `--use_rl` flag to activate the RL training phase.
  - **Warmup Phase**: Training can be configured with `--rl_warmup_steps` to perform standard supervised training for a set number of steps before switching to RL, ensuring model stability.
  - **Simplified RL Step**: A "DPO-style" one-step GRPO update is implemented. Instead of a full, costly rollout, it rewards the model for making the correct denoising decision at a single, random timestep, providing an efficient and effective training signal.
