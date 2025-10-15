import torch
import numpy as np
from diffusers.utils.torch_utils import randn_tensor

def sde_step_with_logprob(scheduler, model_output, timestep, sample, prev_sample=None, generator=None, noise_level=0.0):
    """
    Performs a single step of the SDE scheduler and computes the log probability.
    This version is adapted for ODE-based schedulers like FlowMatchEulerDiscreteScheduler.
    """
    
    # Get the device from the input tensor for the final output
    device = sample.device

    # Indexing tensors must be on the CPU since scheduler.sigmas and scheduler.timesteps are on the CPU.
    timestep_cpu = timestep.to('cpu')
    unique_timesteps_cpu, indices_cpu = torch.unique(timestep_cpu, return_inverse=True)
    
    # Map scheduler timesteps to indices
    timestep_indices_map = {t.item(): i for i, t in enumerate(scheduler.timesteps)}
    
    # Get the indices in the sigmas tensor for the current timesteps (on CPU)
    sigma_indices_cpu = torch.tensor([timestep_indices_map[t.item()] for t in unique_timesteps_cpu])
    
    # Get sigma_t and sigma_t_next using the CPU indices
    sigma_t = scheduler.sigmas[sigma_indices_cpu][indices_cpu]
    
    # Clamp the next indices to avoid out-of-bounds error
    sigma_indices_next_cpu = (sigma_indices_cpu + 1).clamp(max=len(scheduler.sigmas) - 1)
    sigma_t_next = scheduler.sigmas[sigma_indices_next_cpu][indices_cpu]
    
    # Handle the boundary case for the very last timestep, where the next sigma should be 0.
    is_last_step = (sigma_indices_cpu[indices_cpu] == len(scheduler.timesteps) - 1)
    sigma_t_next[is_last_step] = 0.0

    # Move sigmas to the correct device for calculations
    sigma_t = sigma_t.to(device)
    sigma_t_next = sigma_t_next.to(device)
    
    # For ODE schedulers, gamma is typically 0 (no stochastic component)
    gamma_t = torch.zeros_like(sigma_t)
    
    derivative = (sample - model_output) / sigma_t.view(-1, 1, 1, 1)
    dt = (sigma_t_next - sigma_t).view(-1, 1, 1, 1)

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
        # In an ODE step, the next sample is deterministic.
        prev_sample = prev_sample_mean
    
    # compute the log prob of prev_sample given the distribution N(prev_sample_mean, std_dev_t^2 * I)
    # The normalization constant is omitted as it cancels out in the PPO ratio.
    log_prob = (
        -0.5
        * torch.sum(
            ((prev_sample.float() - prev_sample_mean.float()) / (std_dev_t.view(-1, 1, 1, 1) + epsilon)) ** 2,
            dim=[1, 2, 3],
        )
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t
