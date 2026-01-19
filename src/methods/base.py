"""
Base Class for Generative Methods

This module defines the abstract interface that generative methods
(e.g., DDPM) must implement. This ensures consistency across different
implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn as nn


class BaseMethod(nn.Module, ABC):
    """
    Abstract base class for generative modeling methods.

    Methods (e.g., DDPM) should inherit from this class
    and implement the required methods.

    Attributes:
        model: The neural network (typically a U-Net) used for prediction
        device: Device to run computations on
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ):
        """
        Initialize the method.

        Args:
            model: Neural network for prediction (e.g., UNet)
            device: Device to use for computations
        """
        super().__init__()
        self.model = model
        self.device = device

    @abstractmethod
    def compute_loss(
        self,
        x: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the training loss for a batch of data.

        Args:
            x: Clean data samples of shape (batch_size, channels,
               height, width)
            **kwargs: Additional method-specific arguments

        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging
        """
        pass

    @abstractmethod
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate samples from the model.

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments

        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        pass

    def train_mode(self) -> None:
        """Set the model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set the model to evaluation mode."""
        self.model.eval()

    def to(self, *args: Any, **kwargs: Any) -> "BaseMethod":
        """
        Move the method to a device.

        Args:
            *args: Positional arguments (device, dtype, etc.)
            **kwargs: Keyword arguments (device, dtype, non_blocking)

        Returns:
            self for chaining
        """
        super().to(*args, **kwargs)

        # Extract device from args or kwargs
        device = None
        if len(args) > 0 and isinstance(
            args[0],
            (torch.device, str)
        ):
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]

        if device is not None:
            if isinstance(device, str):
                self.device = torch.device(device)
            elif isinstance(device, torch.device):
                self.device = device

        return self
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """
        Return model parameters for optimizer.

        Args:
            recurse: Recursively include submodule parameters

        Returns:
            Iterator over parameters
        """
        return self.model.parameters(recurse=recurse)

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
        if destination is None:
            destination = {}

        destination["model"] = self.model.state_dict()
        return destination

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """
        Load a state dict from a checkpoint.

        Args:
            state_dict: State dict to load
            strict: Strictly enforce matching keys
            assign: Assign state dict to parameters directly

        Returns:
            Incompatible keys
        """
        if "model" in state_dict:
            return self.model.load_state_dict(
                state_dict["model"],
                strict=strict,
                assign=assign,
            )
        return self.model.load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )