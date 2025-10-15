import torch
import numpy as np
from diffusers.utils.torch_utils import randn_tensor

def sde_step_with_logprob(scheduler, model_output, timestep, sample, prev_sample=None, generator=None, noise_level=0.0):
    """
    Performs a single step of the SDE scheduler and computes the log probability.
    This version is adapted for ODE-based schedulers like FlowMatchEulerDiscreteScheduler.
    """
    
    # For ODE schedulers, sigmas are pre-computed. We find the sigma for the current and next timestep.
    # The timestep is a tensor, so we need to use it to index the sigmas tensor.
    # First, find the unique timesteps and their indices
    unique_timesteps, indices = torch.unique(timestep, return_inverse=True)
    
    # Map scheduler timesteps to indices
    timestep_indices = {t.item(): i for i, t in enumerate(scheduler.timesteps)}
    
    # Get the indices in the sigmas tensor for the current timesteps
    sigma_indices = torch.tensor([timestep_indices[t.item()] for t in unique_timesteps], device=timestep.device)
    
    # Get sigma_t and sigma_t_next using the indices
    sigma_t = scheduler.sigmas[sigma_indices][indices]
    sigma_t_next = scheduler.sigmas[sigma_indices + 1][indices]
    
    # For ODE schedulers, gamma is typically 0 (no stochastic component)
    gamma_t = torch.zeros_like(sigma_t)
    
    derivative = (sample - model_output) / sigma_t[:, None, None, None]
    dt = (sigma_t_next - sigma_t)[:, None, None, None]

    prev_sample_mean = sample + derivative * dt

    # The std_dev of the noise term in an ODE step is effectively zero,
    # but for log_prob calculation, we might need a small non-zero value.
    # Let's use a small epsilon or the original logic's gamma * dt.
    # Original: std_dev_t = torch.sqrt(gamma_t * dt) -> would be zero.
    # Let's keep a small constant noise floor for numerical stability in log_prob.
    std_dev_t = torch.sqrt(gamma_t * dt.squeeze()) # This will be a tensor of zeros
    
    # Let's add a very small epsilon to std_dev_t to avoid division by zero in log_prob.
    # This is a common practice when adapting SDE logic to ODEs for RL.
    epsilon = 1e-8
    
    # if prev_sample is not provided, sample it from the distribution
    if prev_sample is None:
        noise = randn_tensor(
            sample.shape,
            generator=generator,
            device=sample.device,
            dtype=sample.dtype,
        )
        # In an ODE step, the next sample is deterministic.
        # prev_sample = prev_sample_mean
        # However, to keep the GRPO structure, we might add a tiny bit of noise.
        # Let's stick to the deterministic path for correctness.
        prev_sample = prev_sample_mean
    
    # compute the log prob of prev_sample given the distribution N(prev_sample_mean, std_dev_t^2 * I)
    # The normalization constant is omitted as it cancels out in the PPO ratio.
    log_prob = (
        -0.5
        * torch.sum(
            ((prev_sample.float() - prev_sample_mean.float()) / (std_dev_t[:, None, None, None] + epsilon)) ** 2,
            dim=[1, 2, 3],
        )
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t
