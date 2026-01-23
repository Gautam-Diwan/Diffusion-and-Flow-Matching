"""
Alternative DDPM Implementation: Predict x₀ instead of noise

Mathematical derivation:
Original DDPM: x_t = √(ᾱ_t) * x₀ + √(1 - ᾱ_t) * ε
              Model predicts: ε_θ(x_t, t)
              Loss: ||ε - ε_θ(x_t, t)||²

Alternative (x₀ prediction):
              Model predicts: x₀_θ(x_t, t) directly
              Loss: ||x₀ - x₀_θ(x_t, t)||²

The reverse process is identical—only the parametrization changes.
"""

import math
from typing import Any, Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPMx0(BaseMethod):
    """DDPM that predicts x₀ (clean image) instead of noise."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Create beta schedule (identical to standard DDPM)
        betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            device=device
        )

        # Precompute useful values
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]]
        )

        # Register buffers (not trainable parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Variance schedule (same as standard DDPM)
        posterior_var = (
            betas * (1.0 - alphas_cumprod_prev) /
            (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_var", posterior_var)
        self.register_buffer(
            "posterior_log_var_clipped",
            torch.log(torch.clamp(posterior_var, min=1e-20))
        )
        self.register_buffer(
            "posterior_mean_coeff1",
            betas * torch.sqrt(alphas_cumprod_prev) /
            (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coeff2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) /
            (1.0 - alphas_cumprod)
        )

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean samples at time t.

        Same as standard DDPM - the forward process is parametrization-agnostic.

        Args:
            x_0: Clean samples (batch_size, channels, height, width)
            t: Timestep indices (batch_size,)
            noise: Optional noise (if None, sample from N(0,1))

        Returns:
            x_t: Noisy samples at time t
            noise: The noise used
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alphas_cumprod = cast(torch.Tensor, self.alphas_cumprod)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - alphas_cumprod[t]
        )

        shape = [-1] + [1] * (len(x_0.shape) - 1)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(shape)
        sqrt_one_minus_alphas_cumprod = (
            sqrt_one_minus_alphas_cumprod.view(shape)
        )

        x_t = (sqrt_alphas_cumprod * x_0 +
               sqrt_one_minus_alphas_cumprod * noise)

        return x_t, noise

    def compute_loss(
        self,
        x: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DDPM loss by predicting x₀.

        KEY DIFFERENCE from standard DDPM:
        - Standard DDPM: model predicts noise ε, loss = ||ε - ε_θ(x_t, t)||²
        - This version: model predicts x₀, loss = ||x₀ - x₀_θ(x_t, t)||²

        Args:
            x: Clean data samples (batch_size, channels, height, width)
            **kwargs: Additional arguments

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics
        """
        batch_size = x.shape[0]

        # Sample random timesteps
        t = torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=x.device
        )

        # Sample noise
        noise = torch.randn_like(x)

        # Forward process: add noise
        x_t, noise = self.forward_process(x, t, noise)

        # **KEY CHANGE**: Predict x₀ instead of noise
        x0_pred = self.model(x_t, t)

        # MSE loss between predicted and true x₀
        loss = F.mse_loss(x0_pred, x)

        metrics: Dict[str, float] = {
            "loss": loss.item(),
            "mse": loss.item(),
        }

        return loss, metrics

    @torch.no_grad()
    def reverse_process(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        One reverse diffusion step using x₀ prediction.

        The reverse process is the same mathematically—we use the posterior
        mean and variance. The only difference is we use the x₀ prediction
        to compute the posterior mean.

        Args:
            x_t: Noisy samples at time t
            t: Timestep indices

        Returns:
            x_prev: Samples at time t-1
        """
        # Predict x₀ from noisy sample
        x0_pred = self.model(x_t, t)
        
        # Clip to valid range for numerical stability
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        # Get posterior coefficients
        coeff1 = cast(torch.Tensor, self.posterior_mean_coeff1)[t]
        coeff2 = cast(torch.Tensor, self.posterior_mean_coeff2)[t]
        
        # Reshape for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1)
        shape = [-1] + [1] * (len(x_t.shape) - 1)
        coeff1 = coeff1.view(shape)
        coeff2 = coeff2.view(shape)
        
        # Posterior mean: coeff1 * x₀_pred + coeff2 * x_t
        # This is the theoretical posterior mean for p(x_{t-1} | x_t, x_0)
        mean = coeff1 * x0_pred + coeff2 * x_t
        
        # Add variance noise if not at final step
        log_var = cast(torch.Tensor, self.posterior_log_var_clipped)[t].view(shape)
        if t[0] > 0:
            variance = torch.exp(log_var)
            z = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(variance) * z
        else:
            x_prev = mean
        
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate samples using reverse diffusion.

        Same as standard DDPM—the sampling process is parametrization-agnostic.

        Args:
            batch_size: Number of samples
            image_shape: Shape (channels, height, width)
            **kwargs: Additional arguments

        Returns:
            samples: Generated samples
        """
        self.eval_mode()

        channels, height, width = image_shape

        # Start with pure noise
        x_t = torch.randn(
            batch_size,
            channels,
            height,
            width,
            device=self.device
        )

        # Reverse diffusion loop
        for t_step in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full(
                (batch_size,),
                t_step,
                device=self.device,
                dtype=torch.long
            )
            x_t = self.reverse_process(x_t, t_batch)

        return x_t

    def to(self, *args: Any, **kwargs: Any) -> "DDPMx0":
        """Move the method to a device."""
        super().to(*args, **kwargs)
        return self

    def state_dict(
        self,
        *,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Get the state dict for checkpointing."""
        state = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        state["num_timesteps"] = self.num_timesteps
        state["beta_start"] = self.beta_start
        state["beta_end"] = self.beta_end
        return state

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: dict,
        device: torch.device
    ) -> "DDPMx0":
        """Create a DDPMx0 instance from config."""
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=int(ddpm_config["num_timesteps"]),
            beta_start=float(ddpm_config["beta_start"]),
            beta_end=float(ddpm_config["beta_end"]),
        )