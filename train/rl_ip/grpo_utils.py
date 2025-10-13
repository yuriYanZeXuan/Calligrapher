import torch

def sde_step_with_logprob(scheduler, pred_original_sample, timestep, sample, prev_sample, generator=None):
    """
    Perform a reverse SDE/ODE step and compute the log probability of the transition.
    This is adapted from the logic in flow_grpo's `sde_step_with_logprob`.

    Args:
        scheduler: The noise scheduler (e.g., FlowMatchEulerDiscreteScheduler).
        pred_original_sample (torch.Tensor): The model's prediction of the original sample x_0.
        timestep (torch.Tensor): The current timestep.
        sample (torch.Tensor): The current noisy sample x_t.
        prev_sample (torch.Tensor): The previous sample x_{t-1}, for which we want to compute the log_prob.
        generator: A torch generator for deterministic noise.

    Returns:
        A tuple of (prev_sample, log_prob, prev_sample_mean, std_dev_t).
    """
    # Get the scheduler's sigma values
    sigma_t = scheduler._get_sigma(timestep)
    sigma_s = scheduler._get_sigma(timestep - 1) # Previous timestep
    
    if sigma_s.sum() < 1e-6: # Corresponds to t=0
        return pred_original_sample, torch.tensor(0.0, device=sample.device), pred_original_sample, torch.tensor(0.0, device=sample.device)

    # Re-calculate the derivative (d) based on Flow Matching ODE
    # d = (x_t - sigma_t * x_0_hat) / sigma_t
    derivative = (sample - sigma_t[:, None, None, None] * pred_original_sample) / sigma_t[:, None, None, None]

    # The mean of the previous sample distribution (Euler method step)
    # x_{t-1}_mean = x_t + (sigma_{t-1} - sigma_t) * d
    dt = sigma_s - sigma_t
    prev_sample_mean = sample + derivative * dt[:, None, None, None]

    # The standard deviation of the transition
    # In the SDE formulation, there's a noise term added at each step.
    # For Euler, the variance is related to (sigma_t^2 - sigma_{t-1}^2)
    # However, for ODEs/Flow Matching, the process is often deterministic.
    # Let's assume a small fixed noise std for stochasticity, or a delta function for deterministic ODEs.
    # For GRPO, we need a distribution to compute log_prob. Let's model it as a Gaussian
    # where the variance is tied to the step size.
    # Variance of the step is (sigma_t^2 - sigma_s^2) * I if going from s to t (forward)
    # The variance of the reverse step is more complex. Let's use the std from the scheduler if available
    # or a small fixed value. A simple approximation for small steps is sqrt(|dt|).
    std_dev_t = torch.sqrt(torch.maximum(sigma_t**2 - sigma_s**2, torch.tensor(1e-8, device=sample.device)))
    
    # Create the distribution for the previous step
    distribution = torch.distributions.Normal(prev_sample_mean.float(), std_dev_t[:, None, None, None].float())

    # Calculate the log probability of the actual `prev_sample` under this distribution
    log_prob = distribution.log_prob(prev_sample).sum(dim=(-1, -2, -1))

    return prev_sample, log_prob, prev_sample_mean, std_dev_t
