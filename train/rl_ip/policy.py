import torch
import torch.nn as nn
from typing import Optional, Tuple
from .grpo_utils import sde_step_with_logprob
from diffusers import FlowMatchEulerDiscreteScheduler

class PolicyWrapper(nn.Module):
    """
    A wrapper for transformer models to make them act as a stochastic policy
    for reinforcement learning. It calculates the log_prob of a transition.
    """
    def __init__(self, transformer, scheduler: FlowMatchEulerDiscreteScheduler, model_type: str = "flux"):
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.model_type = model_type

    def forward(
        self, 
        latents: torch.Tensor,
        prev_latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs one step of the reverse diffusion process, and calculates the log_prob
        of this step.

        Args:
            latents (torch.Tensor): The current noisy latents (x_t).
            prev_latents (torch.Tensor): The latents from the previous step (x_{t-1}), which we want to score.
            timestep (torch.Tensor): The current timestep `t`.
            encoder_hidden_states (torch.Tensor): The text prompt embeddings.
            **kwargs: Additional arguments for the transformer model.

        Returns:
            A tuple containing:
            - prev_sample (torch.Tensor): The sampled previous latent state.
            - log_prob (torch.Tensor): The log probability of `prev_latents` given the current state and action.
            - prev_sample_mean (torch.Tensor): The mean of the distribution for the previous latent state.
            - std_dev_t (torch.Tensor): The standard deviation of the distribution.
        """
        # The transformer predicts the original sample x_0 (or noise, depending on training)
        # Assuming the model predicts x_0, as is common in flow matching.
        pred_original_sample = self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        )[0]

        # Use the utility function to perform the step and get the log_prob
        prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
            self.scheduler,
            pred_original_sample,
            timestep,
            latents,
            prev_latents,
        )
        
        return prev_sample, log_prob, prev_sample_mean, std_dev_t

def create_policy(transformer, scheduler, model_type):
    """Factory function to wrap a transformer model in a policy."""
    return PolicyWrapper(transformer, scheduler, model_type)
