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
from typing import Any, Dict, Tuple
from .base import BaseMethod


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
    def sample(
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