"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Any, Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    """Denoising Diffusion Probabilistic Models."""

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

        # Create beta schedule (linear)
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

        # Variance schedule
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

        # Safe tensor indexing with cast
        alphas_cumprod = cast(torch.Tensor, self.alphas_cumprod)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - alphas_cumprod[t]
        )

        # Reshape for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1)
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
        Compute DDPM training loss (predict noise).

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

        # Predict noise
        noise_pred = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

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
        One reverse diffusion step: x_t -> x_{t-1}

        Args:
            x_t: Noisy samples at time t
            t: Timestep indices

        Returns:
            x_prev: Samples at time t-1
        """
        # Predict noise
        noise_pred = self.model(x_t, t)
    
        # Extract coefficients
        alphas_cumprod = cast(torch.Tensor, self.alphas_cumprod)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t])
        
        # Reshape for broadcasting
        shape = [-1] + [1] * (len(x_t.shape) - 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(shape)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(shape)
        
        # Predict x_0 from x_t and noise
        pred_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)  # Optional: clip to valid range
        
        # Get posterior coefficients
        coeff1 = cast(torch.Tensor, self.posterior_mean_coeff1)[t].view(shape)
        coeff2 = cast(torch.Tensor, self.posterior_mean_coeff2)[t].view(shape)
        
        # Compute posterior mean: coeff1 * x_0 + coeff2 * x_t
        mean = coeff1 * pred_x0 + coeff2 * x_t
        
        # Sample from posterior if not at final step
        log_var = cast(torch.Tensor, self.posterior_log_var_clipped)[t].view(shape)
        if t[0] > 0:
            variance = torch.exp(log_var)
            z = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(variance) * z
        else:
            x_prev = mean
        
        return x_prev
    
    @torch.no_grad()
    def reverse_process_ddim(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        One DDIM reverse step: x_t -> x_{t_prev}
        
        DDIM update rule:
        x_{t-1} = sqrt(α̅_{t-1}) * x̂_0 + sqrt(1-α̅_{t-1}) * ε
        
        where x̂_0 = (x_t - sqrt(1-α̅_t) * ε) / sqrt(α̅_t)
        
        Args:
            x_t: Noisy samples at time t
            t: Current timestep indices
            t_prev: Previous timestep indices
            
        Returns:
            x_prev: Samples at time t_prev
        """
        # Predict noise
        noise_pred = self.model(x_t, t)
        
        # Get alpha values
        alphas_cumprod = cast(torch.Tensor, self.alphas_cumprod)
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t_prev]
        
        # Reshape for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1)
        shape = [-1] + [1] * (len(x_t.shape) - 1)
        alpha_t = alpha_t.view(shape)
        alpha_t_prev = alpha_t_prev.view(shape)
        
        # Predict x_0
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # Clip predicted x_0 to valid range (optional but recommended)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # DDIM update rule
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        sqrt_one_minus_alpha_t_prev = torch.sqrt(1.0 - alpha_t_prev)
        
        x_prev = sqrt_alpha_t_prev * pred_x0 + sqrt_one_minus_alpha_t_prev * noise_pred
        
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int | None = None,
        sampler: str = "ddpm",
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate samples using either DDPM or DDIM sampling.
        
        Args:
            batch_size: Number of samples
            image_shape: Shape (channels, height, width)
            num_steps: Number of sampling steps (for DDIM, can be less than num_timesteps)
            sampler: "ddpm" or "ddim"
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
        
        if sampler == "ddpm":
            # Original DDPM sampling (use all timesteps)
            for t_step in range(self.num_timesteps - 1, -1, -1):
                t_batch = torch.full(
                    (batch_size,),
                    t_step,
                    device=self.device,
                    dtype=torch.long
                )
                x_t = self.reverse_process(x_t, t_batch)
                
        elif sampler == "ddim":
            # DDIM sampling (can use fewer steps)
            if num_steps is None:
                num_steps = self.num_timesteps
                
            # Create timestep subsequence
            # e.g., if num_timesteps=1000 and num_steps=100, use [1000, 990, 980, ..., 10, 0]
            if num_steps >= self.num_timesteps:
                timesteps = list(range(self.num_timesteps - 1, -1, -1))
            else:
                # Subsample timesteps uniformly
                step_size = self.num_timesteps // num_steps
                timesteps = list(range(self.num_timesteps - 1, -1, -step_size))
                # Ensure we end at 0
                if timesteps[-1] != 0:
                    timesteps.append(0)
            
            # DDIM reverse diffusion loop
            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                t_prev = timesteps[i + 1]
                
                t_batch = torch.full(
                    (batch_size,),
                    t,
                    device=self.device,
                    dtype=torch.long
                )
                t_prev_batch = torch.full(
                    (batch_size,),
                    t_prev,
                    device=self.device,
                    dtype=torch.long
                )
                
                x_t = self.reverse_process_ddim(x_t, t_batch, t_prev_batch)
        else:
            raise ValueError(f"Unknown sampler: {sampler}. Choose 'ddpm' or 'ddim'")
        
        return x_t

    def to(self, *args: Any, **kwargs: Any) -> "DDPM":
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
        """
        Get the state dict for checkpointing.

        Args:
            destination: Output dictionary
            prefix: Prefix for keys
            keep_vars: Keep variables instead of tensors

        Returns:
            Dictionary containing method state
        """
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
    ) -> "DDPM":
        """Create a DDPM instance from config."""
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=int(ddpm_config["num_timesteps"]),
            beta_start=float(ddpm_config["beta_start"]),
            beta_end=float(ddpm_config["beta_end"]),
        )