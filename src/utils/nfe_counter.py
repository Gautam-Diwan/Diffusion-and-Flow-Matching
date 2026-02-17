"""
NFE (Number of Function Evaluations) Counter

Wraps a model to count the number of forward passes during sampling.
This is the true cost metric since each forward pass is expensive.
"""

import torch.nn as nn


class NFECounter(nn.Module):
    """
    Wrapper that counts neural network forward passes.
    
    Usage:
        >>> model = UNet(...)
        >>> counter = NFECounter(model)
        >>> output = counter(x, t)  # Increments counter
        >>> print(f"NFE: {counter.nfe}")
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.nfe = 0
    
    def forward(self, *args, **kwargs):
        """Forward pass that increments counter."""
        self.nfe += 1
        return self.model(*args, **kwargs)
    
    def reset(self):
        """Reset counter to zero."""
        self.nfe = 0