"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Any, Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod

# Add these imports at the top
import numpy as np
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
    

    def get_time_steps(self, num_steps: int, skip_type: str = "time_uniform") -> torch.Tensor:
        """
        Generate timestep schedule for sampling.
        
        Args:
            num_steps: Number of sampling steps
            skip_type: "time_uniform" or "logSNR"
            
        Returns:
            Timestep schedule of shape (num_steps + 1,)
        """
        if skip_type == "time_uniform":
            # Uniform in timestep space
            timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1)
        elif skip_type == "logSNR":
            # Uniform in log-SNR space (better for high-order solvers)
            alphas_cumprod = cast(torch.Tensor, self.alphas_cumprod)
            lambda_T = -torch.log(alphas_cumprod[-1] / (1 - alphas_cumprod[-1])) / 2
            lambda_0 = -torch.log(alphas_cumprod[0] / (1 - alphas_cumprod[0])) / 2
            
            logSNR = torch.linspace(lambda_T.item(), lambda_0.item(), num_steps + 1)
            
            # Convert log-SNR back to timesteps
            timesteps = []
            for ls in logSNR:
                # Find timestep t such that lambda_t ≈ ls
                # lambda_t = -log(alpha_t / (1 - alpha_t)) / 2
                # Solve for t
                alpha_t = torch.exp(-2 * ls) / (1 + torch.exp(-2 * ls))
                
                # Find closest timestep
                diff = torch.abs(alphas_cumprod - alpha_t)
                t = torch.argmin(diff)
                timesteps.append(t)
            
            timesteps = torch.tensor(timesteps, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown skip_type: {skip_type}")
        
        return timesteps.to(self.device)

    @torch.no_grad()
    def dpm_solver_first_order_update(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        model_output: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        First-order DPM-Solver update (equivalent to DDIM).
        
        Args:
            x_t: Current samples
            t: Current timestep
            t_prev: Previous timestep
            model_output: Precomputed model output (optional)
            
        Returns:
            x_{t_prev}: Updated samples
        """
        if model_output is None:
            model_output = self.model(x_t, t)
        
        # Get alpha values
        alphas_cumprod = cast(torch.Tensor, self.alphas_cumprod)
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_prev = alphas_cumprod[t_prev].view(-1, 1, 1, 1)
        
        # Compute lambda (log-SNR)
        lambda_t = torch.log(alpha_t) - torch.log(1 - alpha_t)
        lambda_prev = torch.log(alpha_prev) - torch.log(1 - alpha_prev)
        h = lambda_prev - lambda_t
        
        # Predict x_0
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t
        
        # First-order exponential integrator
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
        
        x_prev = sqrt_alpha_prev * x_0_pred + sqrt_one_minus_alpha_prev * (
            model_output + (torch.exp(h) - 1) * model_output
        )
        
        return x_prev

    @torch.no_grad()
    def dpm_solver_second_order_update(
        self,
        x_t: torch.Tensor,
        t_list: list,
        model_output_list: list,
    ) -> torch.Tensor:
        """
        Second-order multistep DPM-Solver update (2M).
        
        Uses model outputs from two previous steps for higher accuracy.
        
        Args:
            x_t: Current samples
            t_list: List of [t_{i-1}, t_i, t_{i+1}] (3 timesteps)
            model_output_list: List of model outputs at [t_{i-1}, t_i]
            
        Returns:
            x_{t_{i+1}}: Updated samples
        """
        assert len(t_list) == 3 and len(model_output_list) == 2
        
        t_prev_prev, t_prev, t = t_list[-3], t_list[-2], t_list[-1]
        model_prev_prev, model_prev = model_output_list[-2], model_output_list[-1]
        
        # Get alpha values
        alphas_cumprod = cast(torch.Tensor, self.alphas_cumprod)
        alpha_prev_prev = alphas_cumprod[t_prev_prev].view(-1, 1, 1, 1)
        alpha_prev = alphas_cumprod[t_prev].view(-1, 1, 1, 1)
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Compute lambda values
        lambda_prev_prev = torch.log(alpha_prev_prev) - torch.log(1 - alpha_prev_prev)
        lambda_prev = torch.log(alpha_prev) - torch.log(1 - alpha_prev)
        lambda_t = torch.log(alpha_t) - torch.log(1 - alpha_t)
        
        h = lambda_t - lambda_prev
        h_prev = lambda_prev - lambda_prev_prev
        r = h_prev / h
        
        # Predict x_0 from previous step
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
        x_0_pred = (x_t - sqrt_one_minus_alpha_prev * model_prev) / sqrt_alpha_prev
        
        # Second-order correction using model_prev_prev
        D_prev_prev = model_prev_prev
        D_prev = model_prev
        
        # Multistep coefficients
        D = D_prev + (1 / (2 * r)) * (D_prev - D_prev_prev)
        
        # Second-order exponential integrator
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        x_t_next = (
            sqrt_alpha_t * x_0_pred +
            sqrt_one_minus_alpha_t * (
                D_prev + (torch.exp(h) - 1 - h) / h * D
            )
        )
        
        return x_t_next

    @torch.no_grad()
    def dpm_solver_third_order_update(
        self,
        x_t: torch.Tensor,
        t_list: list,
        model_output_list: list,
    ) -> torch.Tensor:
        """
        Third-order multistep DPM-Solver update (3M).
        
        Uses model outputs from three previous steps for even higher accuracy.
        
        Args:
            x_t: Current samples
            t_list: List of [t_{i-2}, t_{i-1}, t_i, t_{i+1}] (4 timesteps)
            model_output_list: List of model outputs at [t_{i-2}, t_{i-1}, t_i]
            
        Returns:
            x_{t_{i+1}}: Updated samples
        """
        assert len(t_list) == 4 and len(model_output_list) == 3
        
        t_prev_prev_prev = t_list[-4]
        t_prev_prev = t_list[-3]
        t_prev = t_list[-2]
        t = t_list[-1]
        
        model_prev_prev_prev = model_output_list[-3]
        model_prev_prev = model_output_list[-2]
        model_prev = model_output_list[-1]
        
        # Get alpha values
        alphas_cumprod = cast(torch.Tensor, self.alphas_cumprod)
        alpha_prev_prev_prev = alphas_cumprod[t_prev_prev_prev].view(-1, 1, 1, 1)
        alpha_prev_prev = alphas_cumprod[t_prev_prev].view(-1, 1, 1, 1)
        alpha_prev = alphas_cumprod[t_prev].view(-1, 1, 1, 1)
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Compute lambda values
        lambda_prev_prev_prev = torch.log(alpha_prev_prev_prev) - torch.log(1 - alpha_prev_prev_prev)
        lambda_prev_prev = torch.log(alpha_prev_prev) - torch.log(1 - alpha_prev_prev)
        lambda_prev = torch.log(alpha_prev) - torch.log(1 - alpha_prev)
        lambda_t = torch.log(alpha_t) - torch.log(1 - alpha_t)
        
        h = lambda_t - lambda_prev
        h_prev = lambda_prev - lambda_prev_prev
        h_prev_prev = lambda_prev_prev - lambda_prev_prev_prev
        
        r0 = h_prev / h
        r1 = h_prev_prev / h
        
        # Predict x_0
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
        x_0_pred = (x_t - sqrt_one_minus_alpha_prev * model_prev) / sqrt_alpha_prev
        
        # Third-order correction
        D_prev_prev_prev = model_prev_prev_prev
        D_prev_prev = model_prev_prev
        D_prev = model_prev
        
        # Compute divided differences
        D_1 = (1 / r0) * (D_prev - D_prev_prev)
        D_2 = (1 / (r0 + r1)) * (D_1 - (1 / r1) * (D_prev_prev - D_prev_prev_prev))
        
        # Third-order exponential integrator
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        exp_h = torch.exp(h)
        phi_1 = (exp_h - 1) / h
        phi_2 = (exp_h - 1 - h) / (h ** 2)
        phi_3 = (exp_h - 1 - h - h ** 2 / 2) / (h ** 3)
        
        x_t_next = (
            sqrt_alpha_t * x_0_pred +
            sqrt_one_minus_alpha_t * (
                D_prev +
                h * phi_1 * D_1 +
                h ** 2 * phi_2 * D_2 +
                h ** 3 * phi_3 * D_2  # Simplified third-order term
            )
        )
        
        return x_t_next

    @torch.no_grad()
    def sample_dpm_solver(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 20,
        order: int = 2,
        method: str = "multistep",
        skip_type: str = "time_uniform",
        **kwargs: Any
    ) -> torch.Tensor:
        """
        DPM-Solver++ sampling.
        
        Args:
            batch_size: Number of samples
            image_shape: Shape (C, H, W)
            num_steps: Number of sampling steps (default: 20)
            order: Solver order (1, 2, or 3)
            method: "singlestep" or "multistep"
            skip_type: "time_uniform" or "logSNR"
            
        Returns:
            Generated samples
        """
        assert order in [1, 2, 3], f"Order must be 1, 2, or 3, got {order}"
        assert method in ["singlestep", "multistep"], f"Method must be 'singlestep' or 'multistep'"
        
        self.eval_mode()
        
        channels, height, width = image_shape
        
        # Start with pure noise
        x_t = torch.randn(batch_size, channels, height, width, device=self.device)
        
        # Get timestep schedule
        timesteps = self.get_time_steps(num_steps, skip_type)
        timesteps = timesteps.long()
        
        if method == "singlestep":
            # Single-step method: each step is independent
            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                t_prev = timesteps[i + 1]
                
                t_batch = t.expand(batch_size)
                t_prev_batch = t_prev.expand(batch_size)
                
                if order == 1:
                    x_t = self.dpm_solver_first_order_update(x_t, t_batch, t_prev_batch)
                elif order >= 2:
                    # For higher orders in single-step, use Runge-Kutta-like approach
                    # For simplicity, fall back to first-order
                    x_t = self.dpm_solver_first_order_update(x_t, t_batch, t_prev_batch)
        
        else:  # multistep
            # Store previous model outputs for multistep method
            model_output_list = []
            t_list = []
            
            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                t_next = timesteps[i + 1]
                
                t_batch = t.expand(batch_size)
                t_next_batch = t_next.expand(batch_size)
                
                # Compute model output
                model_output = self.model(x_t, t_batch)
                
                # Store for multistep
                model_output_list.append(model_output)
                t_list.append(t_batch[0])
                
                # Determine which order to use based on available history
                current_order = min(order, len(model_output_list))
                
                if current_order == 1:
                    x_t = self.dpm_solver_first_order_update(
                        x_t, t_batch, t_next_batch, model_output
                    )
                elif current_order == 2:
                    # Need at least 2 previous outputs
                    if len(model_output_list) >= 2:
                        t_list_subset = t_list[-2:] + [t_next_batch[0]]
                        x_t = self.dpm_solver_second_order_update(
                            x_t, t_list_subset, model_output_list[-2:]
                        )
                    else:
                        # Fall back to first-order
                        x_t = self.dpm_solver_first_order_update(
                            x_t, t_batch, t_next_batch, model_output
                        )
                elif current_order == 3:
                    # Need at least 3 previous outputs
                    if len(model_output_list) >= 3:
                        t_list_subset = t_list[-3:] + [t_next_batch[0]]
                        x_t = self.dpm_solver_third_order_update(
                            x_t, t_list_subset, model_output_list[-3:]
                        )
                    else:
                        # Fall back to second-order or first-order
                        if len(model_output_list) >= 2:
                            t_list_subset = t_list[-2:] + [t_next_batch[0]]
                            x_t = self.dpm_solver_second_order_update(
                                x_t, t_list_subset, model_output_list[-2:]
                            )
                        else:
                            x_t = self.dpm_solver_first_order_update(
                                x_t, t_batch, t_next_batch, model_output
                            )
                
                # Keep only necessary history (prevent memory buildup)
                if len(model_output_list) > order:
                    model_output_list.pop(0)
                    t_list.pop(0)
        
        return x_t


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
        Generate samples using DDPM, DDIM, or DPM-Solver++.
        
        Args:
            batch_size: Number of samples
            image_shape: Shape (channels, height, width)
            num_steps: Number of sampling steps
            sampler: "ddpm", "ddim", or "dpm_solver"
            **kwargs: Additional sampler-specific arguments:
                - For DPM-Solver:
                    - order: int (1, 2, or 3) - solver order
                    - method: str ("singlestep" or "multistep")
                    - skip_type: str ("time_uniform" or "logSNR")
            
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
        
        elif sampler == "dpm_solver":
            # DPM-Solver++ sampling
            if num_steps is None:
                num_steps = 20  # Default for DPM-Solver
            
            # Extract DPM-Solver specific kwargs
            order = kwargs.get('order', 2)
            method = kwargs.get('method', 'multistep')
            skip_type = kwargs.get('skip_type', 'time_uniform')
            
            x_t = self.sample_dpm_solver(
                batch_size=batch_size,
                image_shape=image_shape,
                num_steps=num_steps,
                order=order,
                method=method,
                skip_type=skip_type,
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}. Choose 'ddpm', 'ddim', or 'dpm_solver'")
        
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