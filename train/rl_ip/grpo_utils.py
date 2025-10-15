import torch
import numpy as np
from diffusers.utils.torch_utils import randn_tensor

def sde_step_with_logprob(scheduler, model_output, timestep, sample, prev_sample=None, generator=None, noise_level=0.0):
    """
    Performs a single step of the SDE scheduler and computes the log probability,
    aligned with the implementation from flow_grpo.
    """
    
    sigma_t = scheduler.sde_noise.sigma(timestep)
    gamma_t = scheduler.sde_noise.gamma(timestep)
    
    derivative = (sample - model_output) / sigma_t
    dt = scheduler.dt[timestep]

    prev_sample_mean = sample + derivative * dt[:, None, None, None]

    std_dev_t = torch.sqrt(gamma_t * dt)
    
    # if prev_sample is not provided, sample it from the distribution
    if prev_sample is None:
        noise = randn_tensor(
            sample.shape,
            generator=generator,
            device=sample.device,
            dtype=sample.dtype,
        )
        prev_sample = prev_sample_mean + noise_level * noise * std_dev_t[:, None, None, None]
    
    # compute the log prob of prev_sample given the distribution N(prev_sample_mean, std_dev_t^2 * I)
    # The normalization constant is omitted as it cancels out in the PPO ratio.
    log_prob = (
        -0.5
        * torch.sum(
            ((prev_sample.float() - prev_sample_mean.float()) / std_dev_t[:, None, None, None]) ** 2,
            dim=[1, 2, 3],
        )
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t
