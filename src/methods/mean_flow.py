"""
MeanFlow: One-step generative modeling via average velocity.

Trains u_θ(z, r, t) to match the MeanFlow identity:
  u = v - (t-r) * (d/dt)u,  with (d/dt)u = v·∂_z u + ∂_t u (JVP).
1-step sampling: x = noise + u_θ(noise, 0, 1).
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple
from .base import BaseMethod
from src.utils import NFECounter
import time


def _sample_r_t(
    batch_size: int,
    device: torch.device,
    sampler: str = "uniform",
    logit_normal_mu: float = -0.4,
    logit_normal_sigma: float = 1.0,
    pct_r_neq_t: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample (r, t) with r <= t. With probability (1 - pct_r_neq_t) set r = t."""
    if sampler == "uniform":
        t1 = torch.rand(batch_size, device=device)
        t2 = torch.rand(batch_size, device=device)
    elif sampler == "logit_normal":
        raw1 = torch.randn(batch_size, device=device) * logit_normal_sigma + logit_normal_mu
        raw2 = torch.randn(batch_size, device=device) * logit_normal_sigma + logit_normal_mu
        t1 = torch.sigmoid(raw1)
        t2 = torch.sigmoid(raw2)
    else:
        raise ValueError(f"Unknown r_t_sampler: {sampler}")

    r = torch.minimum(t1, t2)
    t = torch.maximum(t1, t2)
    # Ensure t > r for non-degenerate interval (avoid div by zero in model)
    t = torch.where(t > r, t, r + 1e-5)

    # With probability (1 - pct_r_neq_t), set r = t (Flow Matching fallback)
    mask = torch.rand(batch_size, device=device) >= pct_r_neq_t
    r = torch.where(mask, t, r)

    return r, t


class MeanFlow(BaseMethod):
    """
    MeanFlow: train average velocity u_θ(z, r, t) for one-step sampling.
    Convention: t=0 is noise, t=1 is data. 1-step: data = noise + u_θ(noise, 0, 1).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        r_t_sampler: str = "uniform",
        logit_normal_mu: float = -0.4,
        logit_normal_sigma: float = 1.0,
        pct_r_neq_t: float = 0.25,
        loss_adaptive_p: float = 1.0,
        loss_adaptive_c: float = 1e-3,
    ):
        super().__init__(model, device)
        self.r_t_sampler = r_t_sampler
        self.logit_normal_mu = logit_normal_mu
        self.logit_normal_sigma = logit_normal_sigma
        self.pct_r_neq_t = pct_r_neq_t
        self.loss_adaptive_p = loss_adaptive_p
        self.loss_adaptive_c = loss_adaptive_c

    def compute_loss(
        self,
        x: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        MeanFlow loss: u_θ(z_t, r, t) vs u_tgt = v - (t-r)*(v·∂_z u_θ + ∂_t u_θ).
        Uses JVP for (d/dt)u; stop_gradient on u_tgt.
        """
        batch_size = x.shape[0]
        x_1 = x
        x_0 = torch.randn_like(x_1, device=x.device)

        r, t = _sample_r_t(
            batch_size,
            x.device,
            sampler=self.r_t_sampler,
            logit_normal_mu=self.logit_normal_mu,
            logit_normal_sigma=self.logit_normal_sigma,
            pct_r_neq_t=self.pct_r_neq_t,
        )

        t_expanded = t.view(-1, 1, 1, 1)
        r_expanded = r.view(-1, 1, 1, 1)
        z_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        v = x_1 - x_0

        # JVP: (d/dt)u with tangent (v, 0, 1). Model takes (z, r, t).
        def fn(z, r_in, t_in):
            return self.model(z, r_in, t_in)

        # Tangents: v same shape as z; 0 for r, 1 for t (per batch element)
        r_tangent = torch.zeros_like(r, device=x.device)
        t_tangent = torch.ones_like(t, device=x.device)

        u, dudt = torch.func.jvp(
            fn,
            (z_t, r, t),
            (v, r_tangent, t_tangent),
        )

        u_tgt = v - (t - r).view(-1, 1, 1, 1) * dudt
        u_tgt = u_tgt.detach()

        error = u - u_tgt
        mse = (error ** 2).mean()

        if self.loss_adaptive_p != 0:
            # Adaptive weight: 1 / (||error||^2 + c)^p
            err_sq = (error ** 2).sum(dim=(1, 2, 3)) + self.loss_adaptive_c
            w = 1.0 / (err_sq ** self.loss_adaptive_p + 1e-8)
            w = w.view(-1, 1, 1, 1)
            loss = (w.detach() * (error ** 2)).mean()
        else:
            loss = mse

        metrics: Dict[str, float] = {
            "loss": loss.item(),
            "mse": mse.item(),
        }
        return loss, metrics

    @torch.no_grad()
    def _sample_onestep(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        **kwargs: Any
    ) -> torch.Tensor:
        """1-step: x = noise + u_θ(noise, 0, 1)."""
        self.eval_mode()
        channels, height, width = image_shape
        x_0 = torch.randn(batch_size, channels, height, width, device=self.device)
        r = torch.zeros(batch_size, device=self.device)
        t = torch.ones(batch_size, device=self.device)
        u = self.model(x_0, r, t)
        return x_0 + u

    @torch.no_grad()
    def _sample_fewstep(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 4,
        **kwargs: Any
    ) -> torch.Tensor:
        """Few-step: integrate from t=0 to t=1 using z_{t+dt} = z_t + dt * u(z_t, t, t+dt)."""
        self.eval_mode()
        channels, height, width = image_shape
        z = torch.randn(batch_size, channels, height, width, device=self.device)
        times = torch.linspace(0, 1, num_steps + 1, device=self.device)
        for i in range(num_steps):
            t_cur = times[i]
            t_next = times[i + 1]
            dt = t_next - t_cur
            r_b = t_cur.expand(batch_size)
            t_b = t_next.expand(batch_size)
            u = self.model(z, r_b, t_b)
            z = z + dt * u
        return z

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 1,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate samples. num_steps=1 for 1-NFE."""
        nfe_counter = NFECounter(self.model)
        original_model = self.model
        self.model = nfe_counter

        start_time = time.time()
        try:
            if num_steps <= 1:
                samples = self._sample_onestep(batch_size, image_shape, **kwargs)
            else:
                samples = self._sample_fewstep(
                    batch_size, image_shape, num_steps=num_steps, **kwargs
                )
        finally:
            self.model = original_model

        wall_clock_time = time.time() - start_time
        metrics = {
            "nfe": nfe_counter.nfe,
            "wall_clock_time": wall_clock_time,
            "sampler": "meanflow",
            "num_steps": num_steps,
        }
        return samples, metrics

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: dict,
        device: torch.device
    ) -> "MeanFlow":
        """Create MeanFlow from config."""
        method_cfg = config.get("method", config.get("mean_flow", {}))
        return cls(
            model=model,
            device=device,
            r_t_sampler=method_cfg.get("r_t_sampler", "logit_normal"),
            logit_normal_mu=float(method_cfg.get("logit_normal_mu", -0.4)),
            logit_normal_sigma=float(method_cfg.get("logit_normal_sigma", 1.0)),
            pct_r_neq_t=float(method_cfg.get("pct_r_neq_t", 0.25)),
            loss_adaptive_p=float(method_cfg.get("loss_adaptive_p", 1.0)),
            loss_adaptive_c=float(method_cfg.get("loss_adaptive_c", 1e-3)),
        )
