"""
Progressive distillation for Flow Matching (20 -> 10 by default).

The student learns a coarse update over one 10-step interval by matching
two fine teacher updates over 20-step intervals.
"""

import time
import copy
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import BaseMethod
from src.methods.flow_matching import FlowMatching
from src.models import create_model_from_config
from src.utils import NFECounter


class ProgressiveDistillation(BaseMethod):
    """
    Progressive distillation from a fixed teacher to a trainable student.

    Teacher defaults:
      - Flow Matching
      - DPM-Solver order=2, singlestep, time_uniform
      - 20 steps

    Student defaults:
      - 10 steps
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        teacher_model: nn.Module,
        teacher_num_steps: int = 20,
        student_num_steps: int = 10,
        teacher_order: int = 2,
        teacher_method: str = "singlestep",
        teacher_skip_type: str = "time_uniform",
        loss_type: str = "huber",
        huber_delta: float = 0.01,
        charbonnier_eps: float = 1e-3,
        fm_anchor_weight: float = 0.05,
        interval_weight_alpha: float = 0.5,
        feature_l2_weight: float = 0.0,
    ):
        super().__init__(model, device)

        if teacher_num_steps % student_num_steps != 0:
            raise ValueError(
                f"teacher_num_steps ({teacher_num_steps}) must be divisible by "
                f"student_num_steps ({student_num_steps})"
            )
        if teacher_method != "singlestep":
            raise ValueError("This implementation supports teacher_method='singlestep' only.")
        if teacher_order != 2:
            raise ValueError("This implementation is configured for teacher_order=2 only.")
        if loss_type not in {"mse", "huber", "charbonnier"}:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        self.teacher_model = teacher_model.to(device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        self.teacher_num_steps = int(teacher_num_steps)
        self.student_num_steps = int(student_num_steps)
        self.teacher_order = int(teacher_order)
        self.teacher_method = teacher_method
        self.teacher_skip_type = teacher_skip_type
        self.loss_type = loss_type
        self.huber_delta = float(huber_delta)
        self.charbonnier_eps = float(charbonnier_eps)
        self.fm_anchor_weight = float(fm_anchor_weight)
        self.interval_weight_alpha = float(interval_weight_alpha)
        self.feature_l2_weight = float(feature_l2_weight)

        self._teacher_ratio = self.teacher_num_steps // self.student_num_steps
        self._teacher_dt = 1.0 / self.teacher_num_steps
        self._student_dt = 1.0 / self.student_num_steps

    def _to_timestep(self, t: torch.Tensor) -> torch.Tensor:
        return (t.clamp(0, 1) * 999).long()

    def _dpm2_singlestep_update(
        self,
        model: nn.Module,
        z: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        dt_t = torch.full_like(t, dt)
        t_mid = (t + 0.5 * dt_t).clamp(0, 1)
        t_next = (t + dt_t).clamp(0, 1)

        v_t = model(z, self._to_timestep(t))
        z_mid = z + 0.5 * dt_t.view(-1, 1, 1, 1) * v_t
        v_mid = model(z_mid, self._to_timestep(t_mid))
        z_next = z + dt_t.view(-1, 1, 1, 1) * v_mid

        # t_next is retained for parity/readability even though caller tracks time.
        _ = t_next
        return z_next

    def _distill_residual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            per_elem = (pred - target) ** 2
        elif self.loss_type == "huber":
            per_elem = F.huber_loss(pred, target, reduction="none", delta=self.huber_delta)
        else:
            diff = pred - target
            per_elem = torch.sqrt(diff * diff + self.charbonnier_eps * self.charbonnier_eps)
        return per_elem.mean(dim=(1, 2, 3))

    def compute_loss(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Distill one coarse student interval from two teacher micro-steps.
        """
        batch_size = x.shape[0]
        x_1 = x
        x_0 = torch.randn_like(x_1)

        # Pick one coarse interval j in [0, student_num_steps-1] per sample.
        interval_idx = torch.randint(
            low=0,
            high=self.student_num_steps,
            size=(batch_size,),
            device=x.device,
        )
        t_start = interval_idx.float() / self.student_num_steps
        t_start_expanded = t_start.view(-1, 1, 1, 1)

        # Start state on the straight FM path.
        z_start = t_start_expanded * x_1 + (1.0 - t_start_expanded) * x_0

        # Teacher target: compose ratio fine 20-step DPM-2 singlestep updates.
        with torch.no_grad():
            z_teacher = z_start
            t_cur = t_start
            for _ in range(self._teacher_ratio):
                z_teacher = self._dpm2_singlestep_update(
                    self.teacher_model, z_teacher, t_cur, self._teacher_dt
                )
                t_cur = (t_cur + self._teacher_dt).clamp(0, 1)

        # Student coarse update over one 10-step interval.
        z_student = self._dpm2_singlestep_update(self.model, z_start, t_start, self._student_dt)

        per_sample_distill = self._distill_residual_loss(z_student, z_teacher)
        interval_weights = 1.0 + self.interval_weight_alpha * t_start
        distill_loss = (interval_weights * per_sample_distill).mean()

        # Small FM anchor keeps velocity field aligned to source task.
        fm_loss = torch.tensor(0.0, device=x.device)
        if self.fm_anchor_weight > 0:
            t_anchor = torch.rand(batch_size, device=x.device)
            t_anchor_expanded = t_anchor.view(-1, 1, 1, 1)
            x_t = t_anchor_expanded * x_1 + (1.0 - t_anchor_expanded) * x_0
            target_velocity = x_1 - x_0
            pred_velocity = self.model(x_t, self._to_timestep(t_anchor))
            fm_loss = F.mse_loss(pred_velocity, target_velocity)

        loss = distill_loss + self.fm_anchor_weight * fm_loss
        feature_l2 = torch.tensor(0.0, device=x.device)
        if self.feature_l2_weight > 0:
            with torch.no_grad():
                teacher_v = self.teacher_model(z_start, self._to_timestep(t_start))
            student_v = self.model(z_start, self._to_timestep(t_start))
            feature_l2 = F.mse_loss(student_v, teacher_v)
            loss = loss + self.feature_l2_weight * feature_l2

        metrics: Dict[str, float] = {
            "loss": loss.item(),
            "distill_loss": distill_loss.item(),
            "fm_anchor_loss": fm_loss.item(),
            "feature_l2_loss": feature_l2.item(),
            "interval_t_mean": t_start.mean().item(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 10,
        sampler: str = "dpm_solver",
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Sample with the student model using FlowMatching samplers.
        """
        # Delegate sampling logic to FlowMatching implementation while reusing
        # the distilled student model.
        sampler_impl = FlowMatching(self.model, self.device)

        # Keep NFE accounting consistent with existing methods.
        nfe_counter = NFECounter(self.model)
        original_model = self.model
        self.model = nfe_counter
        sampler_impl.model = self.model

        start_time = time.time()
        try:
            if sampler == "dpm_solver":
                order = kwargs.get("order", 2)
                method = kwargs.get("method", "singlestep")
                skip_type = kwargs.get("skip_type", "time_uniform")
                samples = sampler_impl._sample_dpm_solver_impl(
                    batch_size=batch_size,
                    image_shape=image_shape,
                    num_steps=num_steps,
                    order=order,
                    method=method,
                    skip_type=skip_type,
                )
            elif sampler == "heun":
                samples = sampler_impl._sample_heun_impl(
                    batch_size=batch_size,
                    image_shape=image_shape,
                    num_steps=num_steps,
                )
            elif sampler == "euler":
                samples = sampler_impl._sample_euler_impl(
                    batch_size=batch_size,
                    image_shape=image_shape,
                    num_steps=num_steps,
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")
        finally:
            self.model = original_model

        wall_clock_time = time.time() - start_time
        metrics = {
            "nfe": nfe_counter.nfe,
            "wall_clock_time": wall_clock_time,
            "sampler": sampler,
            "num_steps": num_steps,
        }
        return samples, metrics

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: dict,
        device: torch.device,
    ) -> "ProgressiveDistillation":
        teacher_cfg = config.get("teacher", {})
        method_cfg = config.get("method", {})

        teacher_ckpt_path = teacher_cfg.get("checkpoint_path", None)
        if teacher_ckpt_path is None:
            raise ValueError("config.teacher.checkpoint_path is required for progressive distillation.")

        teacher_ckpt = torch.load(teacher_ckpt_path, map_location=device)
        teacher_arch_config = teacher_ckpt.get("config", config)
        teacher_model_override = teacher_cfg.get("model", None)
        if teacher_model_override is not None:
            teacher_arch_config = copy.deepcopy(teacher_arch_config)
            base_teacher_model_cfg = teacher_arch_config.get("model", {})
            teacher_arch_config["model"] = {
                **base_teacher_model_cfg,
                **teacher_model_override,
            }
        teacher_model = create_model_from_config(teacher_arch_config).to(device)

        use_ema = bool(teacher_cfg.get("use_ema", True))
        if use_ema and "ema" in teacher_ckpt and "shadow" in teacher_ckpt["ema"]:
            teacher_state = teacher_model.state_dict()
            shadow = teacher_ckpt["ema"]["shadow"]
            for key in teacher_state:
                if key in shadow and shadow[key].shape == teacher_state[key].shape:
                    teacher_state[key] = shadow[key].to(device)
            teacher_model.load_state_dict(teacher_state)
        else:
            teacher_model.load_state_dict(teacher_ckpt["model"])

        return cls(
            model=model,
            device=device,
            teacher_model=teacher_model,
            teacher_num_steps=int(teacher_cfg.get("num_steps", 20)),
            student_num_steps=int(method_cfg.get("student_num_steps", 10)),
            teacher_order=int(teacher_cfg.get("dpm_solver", {}).get("order", 2)),
            teacher_method=teacher_cfg.get("dpm_solver", {}).get("method", "singlestep"),
            teacher_skip_type=teacher_cfg.get("dpm_solver", {}).get("skip_type", "time_uniform"),
            loss_type=method_cfg.get("loss_type", "huber"),
            huber_delta=float(method_cfg.get("huber_delta", 0.01)),
            charbonnier_eps=float(method_cfg.get("charbonnier_eps", 1e-3)),
            fm_anchor_weight=float(method_cfg.get("fm_anchor_weight", 0.05)),
            interval_weight_alpha=float(method_cfg.get("interval_weight_alpha", 0.5)),
            feature_l2_weight=float(method_cfg.get("feature_l2_weight", 0.0)),
        )
