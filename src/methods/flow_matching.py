"""
Flow Matching Implementation

Flow Matching learns a velocity field to transport noise to data along straight paths.
This is simpler and often more efficient than DDPM.

Key differences from DDPM:
- Predicts velocity field instead of noise
- Uses continuous time t ∈ [0, 1] instead of discrete timesteps
- Simpler training objective
- Deterministic sampling via ODE integration
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple, List
from .base import BaseMethod
from src.utils import NFECounter
import time


class FlowMatching(BaseMethod):
    """
    Flow Matching for image generation.
    
    Training: Learn velocity field v(x_t, t) along straight paths
    Sampling: Integrate ODE from noise to data
    
    Args:
        model: Neural network that predicts velocity field v(x_t, t)
        device: Device to run computations on
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ):
        super().__init__(model, device)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Flow Matching training loss.
        
        Training procedure:
        1. x_1 = data (target)
        2. x_0 ~ N(0, I) (source/noise)
        3. t ~ Uniform(0, 1)
        4. x_t = t * x_1 + (1 - t) * x_0 (straight path interpolation)
        5. Target velocity: v_t = x_1 - x_0 (constant along straight paths)
        6. Loss: MSE(v_θ(x_t, t), v_t)
        
        Args:
            x: Clean data samples (batch_size, channels, height, width)
            **kwargs: Additional arguments
        
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics
        """
        batch_size = x.shape[0]
        
        # x_1 is the data (target)
        x_1 = x
        
        # Sample x_0 from standard normal (source)
        x_0 = torch.randn_like(x_1)
        
        # Sample time t uniformly from [0, 1]
        t = torch.rand(batch_size, device=x.device)
        
        # Reshape t for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1)
        t_expanded = t.view(-1, 1, 1, 1)
        
        # Compute x_t along the straight path: x_t = t * x_1 + (1 - t) * x_0
        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        
        # Target velocity (constant along straight paths): v_t = x_1 - x_0
        target_velocity = x_1 - x_0
        
        # Convert continuous t ∈ [0, 1] to discrete timesteps ∈ [0, 999] for U-Net
        # The U-Net expects integer timesteps for its positional embedding
        timesteps = (t * 999).long()
        
        # Predict velocity
        predicted_velocity = self.model(x_t, timesteps)
        
        # MSE loss
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        
        metrics: Dict[str, float] = {
            "loss": loss.item(),
            "mse": loss.item(),
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def _sample_euler_impl(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 100,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate samples using Euler ODE integration.
        
        Sampling procedure:
        1. Start from x_0 ~ N(0, I) at t=0
        2. For t from 0 to 1 with step dt:
            - Predict velocity: v = v_θ(x_t, t)
            - Euler step: x_{t+dt} = x_t + dt * v
        3. Return x_1 (final image)
        
        Args:
            batch_size: Number of samples to generate
            image_shape: Shape (channels, height, width)
            num_steps: Number of integration steps (default: 100)
            **kwargs: Additional arguments
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        
        channels, height, width = image_shape
        
        # Start from pure noise at t=0
        x_t = torch.randn(
            batch_size,
            channels,
            height,
            width,
            device=self.device
        )
        
        # Time step size
        dt = 1.0 / num_steps
        
        # Euler integration from t=0 to t=1
        for step in range(num_steps):
            # Current time in [0, 1]
            t = step / num_steps
            
            # Convert to integer timestep for U-Net embedding [0, 999]
            timesteps = torch.full(
                (batch_size,),
                int(t * 999),
                device=self.device,
                dtype=torch.long
            )
            
            # Predict velocity field
            velocity = self.model(x_t, timesteps)
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            x_t = x_t + dt * velocity
        
        return x_t
    
    @torch.no_grad()
    def _sample_heun_impl(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 20,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate samples using Heun's method (2nd order).
        
        Heun's method is more accurate than Euler but still efficient.
        It's a 2-stage Runge-Kutta method.
        
        Args:
            batch_size: Number of samples to generate
            image_shape: Shape (channels, height, width)
            num_steps: Number of integration steps (default: 20)
            **kwargs: Additional arguments
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        
        channels, height, width = image_shape
        x_t = torch.randn(
            batch_size,
            channels,
            height,
            width,
            device=self.device
        )
        
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = step / num_steps
            t_next = (step + 1) / num_steps
            
            # Convert to integer timesteps
            t_tensor = torch.full(
                (batch_size,),
                int(t * 999),
                device=self.device,
                dtype=torch.long
            )
            t_next_tensor = torch.full(
                (batch_size,),
                int(t_next * 999),
                device=self.device,
                dtype=torch.long
            )
            
            # Heun's method: K1 and K2 stages
            v1 = self.model(x_t, t_tensor)
            x_pred = x_t + dt * v1
            
            v2 = self.model(x_pred, t_next_tensor)
            
            # Average the slopes
            x_t = x_t + (dt / 2) * (v1 + v2)
        
        return x_t
    
    def _get_time_schedule(
        self,
        num_steps: int,
        skip_type: str = "time_uniform"
    ) -> torch.Tensor:
        """
        Generate continuous time schedule for sampling.
        
        Args:
            num_steps: Number of sampling steps
            skip_type: "time_uniform" or "logSNR"
                - time_uniform: uniform spacing in [0, 1]
                - logSNR: uniform in log-SNR space (better for high-order solvers)
        
        Returns:
            Time schedule of shape (num_steps + 1,) in range [0, 1]
        """
        if skip_type == "time_uniform":
            # Uniform spacing in time
            times = torch.linspace(0, 1, num_steps + 1, device=self.device)
        elif skip_type == "logSNR":
            # Uniform in log-SNR space
            # For Flow Matching with straight paths, SNR ∝ t / (1 - t)
            # log-SNR = log(t) - log(1 - t)
            # We want uniform sampling in this space
            
            # At t=0: log-SNR → -∞
            # At t=1: log-SNR → +∞
            
            # Create uniform schedule in log-SNR space
            # Map [0, 1] to log-SNR range
            eps = 1e-6
            logsnr_start = torch.log(torch.tensor(eps / (1 - eps)))
            logsnr_end = torch.log(torch.tensor((1 - eps) / eps))
            
            logsnr_schedule = torch.linspace(
                logsnr_start.item(),
                logsnr_end.item(),
                num_steps + 1,
                device=self.device
            )
            
            # Convert back to time: solve log(t/(1-t)) = logsnr for t
            # t = exp(logsnr) / (1 + exp(logsnr)) = sigmoid(logsnr)
            times = torch.sigmoid(logsnr_schedule)
        else:
            raise ValueError(f"Unknown skip_type: {skip_type}. Choose 'time_uniform' or 'logSNR'")
        
        return times
    
    @torch.no_grad()
    def _sample_dpm_solver_impl(
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
        Generate samples using DPM-Solver for ODE integration.
        
        DPM-Solver is a high-order ODE solver optimized for diffusion models.
        For Flow Matching, we adapt it to work with continuous time [0, 1]
        and velocity predictions.
        
        Args:
            batch_size: Number of samples to generate
            image_shape: Shape (channels, height, width)
            num_steps: Number of sampling steps (default: 20)
            order: Solver order (1, 2, or 3)
                - 1: First-order (similar to Euler)
                - 2: Second-order (better accuracy)
                - 3: Third-order (best accuracy but slower)
            method: Integration method
                - "singlestep": Each step uses only current step info
                - "multistep": Uses history from previous steps (recommended)
            skip_type: Time schedule
                - "time_uniform": Uniform spacing in time
                - "logSNR": Uniform in log-SNR space (better for high-order)
            **kwargs: Additional arguments
        
        Returns:
            Generated samples of shape (batch_size, *image_shape)
        """
        assert order in [1, 2, 3], f"Order must be 1, 2, or 3, got {order}"
        assert method in ["singlestep", "multistep"], \
            f"Method must be 'singlestep' or 'multistep', got {method}"
        
        self.eval_mode()
        
        channels, height, width = image_shape
        
        # Start from pure noise
        x_t = torch.randn(
            batch_size,
            channels,
            height,
            width,
            device=self.device
        )
        
        # Get time schedule
        times = self._get_time_schedule(num_steps, skip_type)
        
        if method == "singlestep":
            return self._sample_dpm_solver_singlestep(
                x_t, times, order, batch_size, image_shape
            )
        else:  # multistep
            return self._sample_dpm_solver_multistep(
                x_t, times, order, batch_size, image_shape
            )
    
    @torch.no_grad()
    def _sample_dpm_solver_singlestep(
        self,
        x_t: torch.Tensor,
        times: torch.Tensor,
        order: int,
        batch_size: int,
        image_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Single-step DPM-Solver: each step is independent.
        
        Args:
            x_t: Current samples
            times: Time schedule of shape (num_steps + 1,)
            order: Solver order (1, 2, or 3)
            batch_size: Batch size
            image_shape: Image shape (unused, for consistency)
        
        Returns:
            Generated samples
        """
        for i in range(len(times) - 1):
            t = times[i]
            t_next = times[i + 1]
            dt = t_next - t
            
            # Convert continuous time to integer timestep for U-Net
            t_int = int(t.item() * 999)
            t_batch = torch.full(
                (batch_size,),
                t_int,
                device=self.device,
                dtype=torch.long
            )
            
            # Predict velocity at current time
            v_t = self.model(x_t, t_batch)
            
            if order == 1:
                # First-order Euler
                x_t = x_t + dt * v_t
            
            elif order == 2:
                # Second-order: predict midpoint and use that velocity
                t_mid = (t + t_next) / 2
                x_mid = x_t + (dt / 2) * v_t
                
                t_mid_int = int(t_mid.item() * 999)
                t_mid_batch = torch.full(
                    (batch_size,),
                    t_mid_int,
                    device=self.device,
                    dtype=torch.long
                )
                
                v_mid = self.model(x_mid, t_mid_batch)
                x_t = x_t + dt * v_mid
            
            elif order == 3:
                # Third-order RK3 (Kutta's method)
                t_mid = (t + t_next) / 2
                
                # K1 stage
                k1 = v_t
                
                # K2 stage at midpoint
                x_2 = x_t + (dt / 2) * k1
                t_mid_int = int(t_mid.item() * 999)
                t_mid_batch = torch.full(
                    (batch_size,),
                    t_mid_int,
                    device=self.device,
                    dtype=torch.long
                )
                k2 = self.model(x_2, t_mid_batch)
                
                # K3 stage at t_next (predicted with K2)
                x_3 = x_t + dt * (2 * k2 - k1)
                t_next_int = int(t_next.item() * 999)
                t_next_batch = torch.full(
                    (batch_size,),
                    t_next_int,
                    device=self.device,
                    dtype=torch.long
                )
                k3 = self.model(x_3, t_next_batch)
                
                # Final update: (K1 + 4*K2 + K3) / 6
                x_t = x_t + (dt / 6) * (k1 + 4 * k2 + k3)
        
        return x_t
    
    @torch.no_grad()
    def _sample_dpm_solver_multistep(
        self,
        x_t: torch.Tensor,
        times: torch.Tensor,
        order: int,
        batch_size: int,
        image_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Multi-step DPM-Solver: uses history from previous steps.
        
        This is more accurate than single-step for higher orders.
        
        Args:
            x_t: Current samples
            times: Time schedule of shape (num_steps + 1,)
            order: Solver order (1, 2, or 3)
            batch_size: Batch size
            image_shape: Image shape (unused, for consistency)
        
        Returns:
            Generated samples
        """
        # Store velocity history for multistep
        velocity_history: List[torch.Tensor] = []
        time_history: List[float] = []
        
        for i in range(len(times) - 1):
            t = times[i]
            t_next = times[i + 1]
            dt = t_next - t
            
            # Convert continuous time to integer timestep
            t_int = int(t.item() * 999)
            t_batch = torch.full(
                (batch_size,),
                t_int,
                device=self.device,
                dtype=torch.long
            )
            
            # Predict velocity at current time
            v_t = self.model(x_t, t_batch)
            velocity_history.append(v_t)
            time_history.append(t.item())
            
            # Determine order based on available history
            current_order = min(order, len(velocity_history))
            
            if current_order == 1:
                # First-order: Euler
                x_t = x_t + dt * v_t
            
            elif current_order == 2:
                # Second-order: linear extrapolation
                if len(velocity_history) >= 2:
                    v_prev = velocity_history[-2]
                    t_prev = time_history[-2]
                    dt_prev = t - torch.tensor(t_prev, device=self.device)
                    
                    # Estimate velocity change
                    dv_dt = (v_t - v_prev) / (dt_prev + 1e-8)
                    
                    # Linear extrapolation
                    v_extrap = v_t + dv_dt * dt
                    x_t = x_t + dt * v_extrap
                else:
                    x_t = x_t + dt * v_t
            
            elif current_order == 3:
                # Third-order: quadratic extrapolation
                if len(velocity_history) >= 3:
                    v_t_2 = velocity_history[-3]
                    v_t_1 = velocity_history[-2]
                    v_t = velocity_history[-1]
                    
                    t_2 = time_history[-3]
                    t_1 = time_history[-2]
                    t = time_history[-1]
                    
                    # Divided differences for quadratic fit
                    dt_1 = t_1 - t_2
                    dt = t - t_1
                    
                    # Second divided difference
                    d1 = (v_t_1 - v_t_2) / (dt_1 + 1e-8)
                    d2 = (v_t - v_t_1) / (dt + 1e-8)
                    dd = (d2 - d1) / (t - t_2 + 1e-8)
                    
                    # Quadratic extrapolation
                    dt_next = times[i + 1].item() - t
                    v_extrap = v_t + d2 * dt_next + dd * dt_next * (dt_next + dt)
                    x_t = x_t + dt_next * v_extrap
                else:
                    # Fall back to lower order
                    if len(velocity_history) >= 2:
                        v_prev = velocity_history[-2]
                        t_prev = time_history[-2]
                        dt_prev = t - torch.tensor(t_prev, device=self.device)
                        dv_dt = (v_t - v_prev) / (dt_prev + 1e-8)
                        v_extrap = v_t + dv_dt * dt
                        x_t = x_t + dt * v_extrap
                    else:
                        x_t = x_t + dt * v_t
            
            # Keep only necessary history
            if len(velocity_history) > order:
                velocity_history.pop(0)
                time_history.pop(0)
        
        return x_t
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 20,
        sampler: str = "euler",
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate samples and return metrics."""
        
        # Wrap model with NFE counter
        nfe_counter = NFECounter(self.model)
        original_model = self.model
        self.model = nfe_counter
        
        start_time = time.time()
        
        try:
            if sampler == "euler":
                samples = self._sample_euler_impl(batch_size, image_shape, num_steps, **kwargs)
            elif sampler == "heun":
                samples = self._sample_heun_impl(batch_size, image_shape, num_steps, **kwargs)
            elif sampler == "dpm_solver":
                order = kwargs.get("order", 2)
                method = kwargs.get("method", "multistep")
                skip_type = kwargs.get("skip_type", "time_uniform")
                samples = self._sample_dpm_solver_impl(
                    batch_size, image_shape, num_steps, order, method, skip_type
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")
        finally:
            # Restore original model
            self.model = original_model
        
        wall_clock_time = time.time() - start_time
        
        metrics = {
            'nfe': nfe_counter.nfe,
            'wall_clock_time': wall_clock_time,
            'sampler': sampler,
            'num_steps': num_steps,
        }
        
        return samples, metrics
    
    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: dict,
        device: torch.device
    ) -> "FlowMatching":
        """Create a FlowMatching instance from config."""
        return cls(
            model=model,
            device=device,
        )