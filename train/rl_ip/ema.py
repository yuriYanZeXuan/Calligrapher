# EMA (Exponential Moving Average) module for model parameters
# Copied and adapted from flow_grpo

from collections.abc import Iterable
import torch


class EMAModuleWrapper:
    """
    Exponential Moving Average (EMA) wrapper for model parameters.
    This helps stabilize training by maintaining a smoothed version of the model weights.
    """
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: torch.device | None = None,
    ):
        """
        Args:
            parameters: Iterable of model parameters to track with EMA
            decay: Decay rate for EMA (higher = slower updates)
            update_step_interval: Update EMA every N steps
            device: Device to store EMA parameters
        """
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]
        
        self.temp_stored_parameters = None
        
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step) -> float:
        """
        Get adaptive decay rate that starts lower and increases to target decay.
        This helps at the beginning of training.
        """
        return min(
            (1 + optimization_step) / (10 + optimization_step),
            self.decay
        )

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step):
        """
        Update EMA parameters using current model parameters.
        
        Args:
            parameters: Current model parameters
            optimization_step: Current training step
        """
        parameters = list(parameters)
        
        one_minus_decay = 1 - self.get_current_decay(optimization_step)
        
        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
                if parameter.requires_grad:
                    if ema_parameter.device == parameter.device:
                        ema_parameter.add_(one_minus_decay * (parameter - ema_parameter))
                    else:
                        # In place calculations to save memory
                        parameter_copy = parameter.detach().to(ema_parameter.device)
                        parameter_copy.sub_(ema_parameter)
                        parameter_copy.mul_(one_minus_decay)
                        ema_parameter.add_(parameter_copy)
                        del parameter_copy

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """Move EMA parameters to specified device/dtype."""
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    @torch.no_grad()
    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True) -> None:
        """
        Copy EMA parameters to model parameters.
        Optionally store current parameters for later restoration.
        
        Args:
            parameters: Model parameters to update
            store_temp: Whether to store current parameters before updating
        """
        if store_temp:
            self.temp_stored_parameters = [parameter.detach().cpu() for parameter in parameters]
        
        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            parameter.data.copy_(ema_parameter.to(parameter.device).data)

    @torch.no_grad()
    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Restore temporarily stored parameters.
        This is used to restore model parameters after using EMA for evaluation.
        
        Args:
            parameters: Model parameters to restore
        """
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters, strict=True):
            parameter.data.copy_(temp_parameter.to(parameter.device))
        
        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters")
        self.to(self.device)

    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }

