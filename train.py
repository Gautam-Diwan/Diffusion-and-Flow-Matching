"""
Training Script for DDPM (Denoising Diffusion Probabilistic Models)

This script provides a training loop for DDPM.
It supports:
- Mixed precision training (AMP)
- Exponential Moving Average (EMA)
- Gradient clipping
- Periodic checkpointing
- MultiGPU training
- Periodic sampling to check the generation quality
- Logging to console and optionally wandb

What you need to implement:
- Incorporate your sampling scheme to this pipeline
- Save generated samples as images for logging

Usage:
    # Train DDPM
    python train.py --method ddpm --config configs/ddpm.yaml

    # Resume training
    python train.py --method ddpm --config configs/ddpm.yaml --resume checkpoints/ddpm_50000.pt
"""

import os
import sys
import argparse
import math
import time
import copy
from datetime import datetime
from typing import Optional, Dict, Any

import yaml
import torch
torch.set_float32_matmul_precision("high")  # best default on Ada/L40S
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.models import UNet, create_model_from_config, create_meanflow_model_from_config
from src.data import create_dataloader_from_config, save_image, unnormalize
from src.methods import DDPM
from src.methods.flow_matching import FlowMatching
from src.methods.mean_flow import MeanFlow
from src.methods.progressive_distillation import ProgressiveDistillation
from src.utils import EMA

import wandb
from PIL import Image as PILImage

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_with_override(config: dict, model_override: Optional[dict] = None) -> nn.Module:
    """Create UNet from config, optionally overriding model sub-config fields."""
    if model_override is None:
        return create_model_from_config(config)
    merged = copy.deepcopy(config)
    merged["model"] = {
        **merged.get("model", {}),
        **model_override,
    }
    return create_model_from_config(merged)


def load_pretrained_fm_weights(
    fm_checkpoint_path: str,
    mf_model: nn.Module,
    device: torch.device,
) -> None:
    """Transfer Flow Matching EMA weights into a MeanFlow model.

    Loads the EMA shadow parameters from an FM checkpoint and copies every
    weight whose key name and shape match the MeanFlow model.  The only FM
    key that has no counterpart is ``time_pos_emb`` (learned embedding table);
    MeanFlow replaces it with parameter-free sinusoidal embeddings so the key
    simply won't appear in MeanFlow's state dict and is skipped automatically.
    """
    checkpoint = torch.load(fm_checkpoint_path, map_location=device)

    if "ema" in checkpoint and "shadow" in checkpoint["ema"]:
        fm_weights = checkpoint["ema"]["shadow"]
        source_label = "EMA shadow"
    else:
        fm_weights = checkpoint.get("model", checkpoint)
        source_label = "model"

    mf_state = mf_model.state_dict()
    transferred, skipped = [], []

    for key in mf_state:
        if key in fm_weights and fm_weights[key].shape == mf_state[key].shape:
            mf_state[key] = fm_weights[key].to(device)
            transferred.append(key)
        else:
            skipped.append(key)

    mf_model.load_state_dict(mf_state)

    print(f"Pretrained init from {source_label} weights ({fm_checkpoint_path}):")
    print(f"  Transferred {len(transferred)}/{len(mf_state)} parameters")
    if skipped:
        print(f"  Skipped (no match): {skipped}")


def setup_logging(config: dict, method_name: str) -> tuple[str, Any]:
    """Set up logging directories and wandb. Returns (log_dir, wandb_run)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config['logging']['dir'], f"{method_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    print(f"Logging to: {log_dir}")

    # Initialize wandb if enabled
    wandb_run = None
    wandb_config = config['logging'].get('wandb', {})
    if wandb_config.get('enabled', False):
        try:
            wandb_run = wandb.init(
                project=wandb_config.get('project', 'cmu-10799-diffusion'),
                entity=wandb_config.get('entity', None),
                name=f"{method_name}_{timestamp}",
                config=config,
                dir=log_dir,
                tags=[method_name],
            )
            print(f"Weights & Biases: {wandb_run.url}")
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")

    return log_dir, wandb_run


def get_distributed_context() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from torchrun env vars if set."""
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed(is_distributed: bool) -> None:
    """Tear down the process group if initialized."""
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module if wrapped by DDP."""
    base = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    # torch.compile wraps modules in OptimizedModule and stores original as _orig_mod.
    if hasattr(base, "_orig_mod"):
        return base._orig_mod
    return base


def reduce_metrics(
    metrics: Dict[str, Any],
    device: torch.device,
    world_size: int,
) -> Dict[str, float]:
    """Average metrics across ranks for consistent logging."""
    if world_size < 2 or not dist.is_initialized():
        return {
            k: (v.detach().item() if torch.is_tensor(v) else float(v))
            for k, v in metrics.items()
        }

    reduced: Dict[str, float] = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            tensor = v.detach()
        else:
            tensor = torch.tensor(v, dtype=torch.float32)
        if tensor.device != device:
            tensor = tensor.to(device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / world_size
        reduced[k] = tensor.item()
    return reduced


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    training_config = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config['learning_rate']),
        betas=tuple(float(b) for b in training_config['betas']),
        weight_decay=float(training_config['weight_decay']),
    )
    return optimizer


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMA],
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: dict,
):
    """Save training checkpoint."""
    model_to_save = unwrap_model(model)
    checkpoint = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'config': config,
    }
    if ema is not None:
        checkpoint['ema'] = ema.state_dict()
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMA],
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> int:
    """Load training checkpoint and return the step."""
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if ema is not None and 'ema' in checkpoint:
        ema.load_state_dict(checkpoint['ema'])
    scaler.load_state_dict(checkpoint['scaler'])
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    step = checkpoint['step']
    print(f"Loaded checkpoint from {path} at step {step}")
    return step


@torch.no_grad()
def generate_samples(
    method_obj,
    num_samples: int,
    image_shape: tuple,
    device: torch.device,
    method_name: str,
    config: dict,
    ema: Optional[EMA] = None,
    current_step: Optional[int] = None,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Generate samples using EMA parameters if available.
    """
    method_obj.eval_mode()
    
    # Track if we successfully applied EMA
    using_ema = False
    
    # Logic: Use EMA if it exists AND we are past the start threshold
    if ema is not None:
        ema_start = config['training'].get('ema_start', 0)
        # If current_step is None, we assume this is a standalone eval (so use EMA)
        # If current_step is set, we check if we've passed the warmup
        if current_step is None or current_step >= ema_start:
            print(f"Sampling with EMA weights...")
            ema.apply_shadow()  # Swap model weights with EMA weights
            using_ema = True

    try:
        # Run sampling (now using EMA weights if applied)
        result = method_obj.sample(
            batch_size=num_samples,
            image_shape=image_shape,
            **sampling_kwargs
        )
    finally:
        # CRITICAL: Always restore original weights to continue training correctly
        if using_ema:
            ema.restore()  # Restore original training weights

    method_obj.train_mode()
    # method.sample returns (samples, metrics); return only the tensor
    if isinstance(result, tuple):
        result = result[0]
    return result

def save_samples(
    samples: torch.Tensor,
    save_path: str,
    num_samples: int,
) -> None:
    """
    Save generated samples as images.

    Args:
        samples: Generated samples tensor with shape (num_samples, C, H, W).
        save_path: File path to save the image grid.
        num_samples: Number of samples, used to calculate grid layout.
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Samples are in [-1, 1] range, save_image handles this
    # Calculate grid layout based on number of samples
    nrow = int(math.ceil(math.sqrt(num_samples)))
    
    # Save the image grid
    save_image(samples, save_path, nrow=nrow)


def train(
    method_name: str,
    config: dict,
    resume_path: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    overfit_single_batch: bool = False,
):
    """
    Main training loop.

    Args:
        method_name: 'ddpm' (you can add more later)
        config: Configuration dictionary
        resume_path: Path to checkpoint to resume from
        pretrained_path: Path to pretrained FM checkpoint for MeanFlow weight init
        overfit_single_batch: If True, train on a single batch repeatedly for debugging
    """
    # Auto-detect distributed setup from environment
    rank, world_size, local_rank = get_distributed_context()

    # Check if config wants distributed training
    config_device = config['infrastructure'].get('device', 'cuda')
    config_num_gpus = config['infrastructure'].get('num_gpus', None)

    # Distributed only if: world_size > 1 AND config allows it
    # Config disables distributed if: device='cpu' OR num_gpus=1
    config_allows_distributed = (
        config_device != 'cpu' and
        (config_num_gpus is None or config_num_gpus > 1)
    )
    is_distributed = world_size > 1 and config_allows_distributed
    is_main_process = rank == 0

    # Distributed training requires CUDA
    if is_distributed and (not torch.cuda.is_available() or config_device == 'cpu'):
        raise RuntimeError(
            "Distributed training requires CUDA. Either:\n"
            "  1. Run without torchrun for CPU training, or\n"
            "  2. Set infrastructure.device='cuda' in config and ensure GPUs are available"
        )

    # Determine device
    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        backend = 'nccl'
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method='env://')
    else:
        use_cuda = torch.cuda.is_available() and config_device != 'cpu'
        device = torch.device('cuda' if use_cuda else 'cpu')

    if is_main_process:
        print("=" * 60)
        print("DEVICE CONFIGURATION")
        print("=" * 60)
        if is_distributed:
            print(f"✓ Distributed training enabled")
            print(f"  - World size: {world_size} GPU(s)")
            print(f"  - Backend: {backend}")
            print(f"  - Device: {device}")
        else:
            if device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name(device)
                print(f"✓ Single GPU training")
                print(f"  - Device: {device} ({gpu_name})")
            else:
                print(f"✓ CPU training")
                print(f"  - Device: {device}")
        print(f"  - Config device setting: {config_device}")
        print(f"  - Mixed precision: {config['infrastructure'].get('mixed_precision', False)}")
        print(f"  - Compile model: {config['infrastructure'].get('compile_model', False)}")
        print(f"  - Channels-last: {config['infrastructure'].get('channels_last', False)}")
        print("=" * 60)

    # Set seed for reproducibility
    seed = int(config['infrastructure']['seed'])
    torch.manual_seed(seed)  # Same seed for all ranks for model init
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    training_config = config['training']
    data_config = config['data']

    # Create data loader
    if is_main_process:
        print("Creating data loader...")
    dataloader = create_dataloader_from_config(config, split='train')
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataloader.dataset,
            batch_size=int(training_config['batch_size']),
            sampler=sampler,
            num_workers=int(data_config['num_workers']),
            pin_memory=data_config['pin_memory'],
            drop_last=True,
        )

    if is_main_process:
        print(f"Dataset size: {len(dataloader.dataset)}")
        print(f"Batches per epoch: {len(dataloader)}")

    # Create model
    if is_main_process:
        print("Creating model...")
    if method_name == 'mean_flow':
        base_model = create_meanflow_model_from_config(config).to(device)
    elif method_name == 'progressive_distillation':
        student_override = config.get('student_model', None)
        base_model = create_model_with_override(config, student_override).to(device)
    else:
        base_model = create_model_from_config(config).to(device)

    use_channels_last = bool(
        config.get("infrastructure", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if use_channels_last:
        base_model = base_model.to(memory_format=torch.channels_last)

    compile_model = bool(config.get("infrastructure", {}).get("compile_model", False))
    compile_mode = str(config.get("infrastructure", {}).get("compile_mode", "reduce-overhead"))
    if compile_model and hasattr(torch, "compile"):
        if is_main_process:
            print(f"Compiling model with torch.compile(mode='{compile_mode}')...")
        try:
            base_model = torch.compile(base_model, mode=compile_mode)
        except Exception as e:
            if is_main_process:
                print(f"Warning: torch.compile failed, continuing without compile: {e}")

    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)

    model = base_model
    if is_distributed:
        # Ensure model is on correct device before DDP wrapping
        assert device.type == 'cuda', f"Rank {rank}: Distributed training requires CUDA, but got device.type={device.type}"
        model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    # Transfer pretrained FM weights into MeanFlow model (before optimizer/EMA)
    if pretrained_path is not None and method_name == 'mean_flow':
        if is_main_process:
            print(f"Loading pretrained FM weights from {pretrained_path}...")
        load_pretrained_fm_weights(pretrained_path, base_model, device)

    num_params = sum(p.numel() for p in base_model.parameters())
    if is_main_process:
        print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Create method
    if is_main_process:
        print(f"Creating {method_name}...")
    if method_name == 'ddpm':
        method = DDPM.from_config(model, config, device)
    elif method_name == 'ddpmx0':
        from src.methods.ddpmx0 import DDPMx0
        method = DDPMx0.from_config(model, config, device)
    elif method_name == 'flow_matching':
        method = FlowMatching.from_config(model, config, device)
    elif method_name == 'mean_flow':
        method = MeanFlow.from_config(model, config, device)
    elif method_name == 'progressive_distillation':
        method = ProgressiveDistillation.from_config(model, config, device)
    else:
        raise ValueError(
            f"Unknown method: {method_name}. "
            "Supported: ddpm, ddpmx0, flow_matching, mean_flow, progressive_distillation."
        )

    # Create optimizer
    optimizer = create_optimizer(model, config) # default to AdamW optimizer

    # Create learning rate scheduler
    # Cosine annealing with warmup is standard for diffusion models
    num_iterations = int(training_config['num_iterations'])
    warmup_iters = int(training_config.get('warmup_iters', 5000))  # 5% warmup by default
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_iterations - warmup_iters,
        eta_min=1e-6
    )

    # Create EMA
    ema = EMA(unwrap_model(model), decay=float(config['training']['ema_decay']))

    # Create gradient scaler for mixed precision
    # Determine device type for GradScaler (cuda or cpu)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = GradScaler(device_type, enabled=config['infrastructure']['mixed_precision'])

    # Setup logging
    log_dir = None
    wandb_run = None
    if is_main_process:
        log_dir, wandb_run = setup_logging(config, method_name)

    # Log model info to wandb
    if is_main_process and wandb_run is not None:
        try:
            wandb.log({
                'model/parameters': num_params,
                'model/parameters_millions': num_params / 1e6,
                'data/dataset_size': len(dataloader.dataset),
                'data/batches_per_epoch': len(dataloader),
            }, step=0)
            # Watch model gradients and parameters
            # wandb.watch(model, log='all', log_freq=config['training']['log_every'])
        except Exception as e:
            print(f"Warning: Failed to log model info to wandb: {e}")


    # Resume from checkpoint if specified
    start_step = 0
    if resume_path is not None:
        # Barrier to ensure all processes wait before loading checkpoint
        if is_distributed:
            dist.barrier()
        start_step = load_checkpoint(resume_path, model, optimizer, ema, scaler, scheduler, device)
    
    # Training config
    num_iterations = int(training_config['num_iterations'])
    log_every = int(training_config['log_every'])
    sample_every = int(training_config['sample_every'])
    save_every = int(training_config['save_every'])
    num_samples = int(training_config['num_samples'])
    gradient_clip_norm = float(training_config['gradient_clip_norm'])
    
    # Image shape for sampling
    image_shape = (int(data_config['channels']), int(data_config['image_size']), int(data_config['image_size']))
    
    # Training loop
    if is_main_process:
        print(f"\nStarting training from step {start_step}...")
        print(f"Total iterations: {num_iterations}")
        if overfit_single_batch:
            print("DEBUG MODE: Overfitting to a single batch")
        print("-" * 50)

    method.train_mode()
    data_iter = iter(dataloader)
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)

    # Pro tips: before big training runs, it's usually a good idea to sanity check 
    # by overfitting to a single batch with a small number of training iterations
    # For single batch overfitting, grab one batch and reuse it
    single_batch = None
    single_batch_base = None  # Store the original small batch
    if overfit_single_batch:
        single_batch_base = next(data_iter)
        if isinstance(single_batch_base, (tuple, list)):
            single_batch_base = single_batch_base[0]  # Handle (image, label) tuples
        if use_channels_last and single_batch_base.dim() == 4:
            single_batch_base = single_batch_base.to(device, memory_format=torch.channels_last)
        else:
            single_batch_base = single_batch_base.to(device)

        # Replicate to match desired batch size
        base_batch_size = single_batch_base.shape[0]
        desired_batch_size = int(training_config['batch_size'])

        if desired_batch_size > base_batch_size:
            # Replicate the batch to reach desired size
            num_repeats = (desired_batch_size + base_batch_size - 1) // base_batch_size
            single_batch = single_batch_base.repeat(num_repeats, 1, 1, 1)[:desired_batch_size]
            if is_main_process:
                print(f"Cached single batch: {base_batch_size} samples replicated to {desired_batch_size}")
                print(f"  Base batch shape: {single_batch_base.shape}")
                print(f"  Training batch shape: {single_batch.shape}")
        else:
            single_batch = single_batch_base
            if is_main_process:
                print(f"Cached single batch with shape: {single_batch.shape}")

    metrics_sum = {}
    metrics_count = 0
    start_time = time.time()

    pbar = tqdm(
        range(start_step, num_iterations),
        initial=start_step,
        total=num_iterations,
        disable=not is_main_process,
    )
    for step in pbar:
        # Get batch (cycle through dataset or use single batch)
        if overfit_single_batch:
            batch = single_batch
        else:
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if isinstance(batch, (tuple, list)):
                batch = batch[0]  # Handle (image, label) tuples

            if use_channels_last and batch.dim() == 4:
                batch = batch.to(device, memory_format=torch.channels_last)
            else:
                batch = batch.to(device)
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        with autocast(device_type, enabled=config['infrastructure']['mixed_precision']):
            loss, metrics = method.compute_loss(batch)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Learning rate schedule step (after warmup)
        if step >= warmup_iters:
            scheduler.step()

        # EMA update - only after ema_start
        ema_start = int(config['training'].get('ema_start', 0))
        if step >= ema_start:
            ema.update()
        
        # Accumulate metrics (store raw values, will reduce when logging)
        for k, v in metrics.items():
            if k not in metrics_sum:
                metrics_sum[k] = []
            metrics_sum[k].append(v.detach().item() if torch.is_tensor(v) else float(v))
        metrics_count += 1
        
        # Logging
        if (step + 1) % log_every == 0:
            elapsed = time.time() - start_time
            steps_per_sec = metrics_count / elapsed

            # Average metrics locally first
            local_avg_metrics = {k: sum(v) / len(v) for k, v in metrics_sum.items()}

            # Reduce metrics across all ranks
            avg_metrics = reduce_metrics(local_avg_metrics, device, world_size)

            # Update progress bar
            if is_main_process:
                pbar.set_postfix({
                    'loss': f"{avg_metrics['loss']:.4f}",
                    'steps/s': f"{steps_per_sec:.2f}",
                })

            # Log to wandb
            if is_main_process and wandb_run is not None:
                log_dict = {
                    'train/step': step + 1,
                    'train/steps_per_sec': steps_per_sec,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                }
                # Add all metrics
                for k, v in avg_metrics.items():
                    log_dict[f'train/{k}'] = v

                try:
                    wandb.log(log_dict, step=step + 1)
                except Exception as e:
                    print(f"Warning: Failed to log to wandb: {e}")


            # Reset metrics
            metrics_sum = {}
            metrics_count = 0
            start_time = time.time()

        # Generate samples
        if (step + 1) % sample_every == 0:
            if is_main_process:
                print(f"\nGenerating samples at step {step + 1}...")
                
                # Build sampling kwargs (use sampler_name to avoid overwriting DataLoader sampler)
                sampling_kwargs = {}
                sampling_config = config.get('sampling', {})
                sampler_name = sampling_config.get('sampler', 'ddpm')
                sampling_kwargs['num_steps'] = sampling_config.get('num_steps', 20)
                sampling_kwargs['sampler'] = sampler_name

                if sampler_name == 'dpm_solver':
                    dpm_config = sampling_config.get('dpm_solver', {})
                    sampling_kwargs['order'] = dpm_config.get('order', 2)
                    sampling_kwargs['method'] = dpm_config.get('method', 'multistep')
                    sampling_kwargs['skip_type'] = dpm_config.get('skip_type', 'time_uniform')

                samples = generate_samples(
                    method,
                    num_samples,
                    image_shape,
                    device,
                    method_name,
                    config,
                    ema,
                    current_step=step + 1,
                    **sampling_kwargs
                )
                sample_path = os.path.join(log_dir, 'samples', f'samples_{step + 1:07d}.png')
                save_samples(samples, sample_path, num_samples)

                # Log samples to wandb
                if wandb_run is not None:
                    try:
                        # Load the saved image and log it
                        img = PILImage.open(sample_path)
                        wandb.log({
                            'samples': wandb.Image(img, caption=f'Step {step + 1}')
                        }, step=step + 1)
                    except Exception as e:
                        print(f"Warning: Failed to log samples to wandb: {e}")

            # Barrier to synchronize all processes after sampling
            if is_distributed:
                dist.barrier()

        # Save checkpoint
        if (step + 1) % save_every == 0:
            if is_main_process:
                checkpoint_path = os.path.join(log_dir, 'checkpoints', f'{method_name}_{step + 1:07d}.pt')
                save_checkpoint(checkpoint_path, model, optimizer, ema, scaler, scheduler, step + 1, config)

            # Barrier to synchronize all processes after checkpoint save
            if is_distributed:
                dist.barrier()
    
    # Save final checkpoint
    if is_main_process:
        final_path = os.path.join(log_dir, 'checkpoints', f'{method_name}_final.pt')
        save_checkpoint(final_path, model, optimizer, ema, scaler, scheduler, num_iterations, config)

        print("\nTraining complete!")
        print(f"Final checkpoint: {final_path}")
        print(f"Samples saved to: {os.path.join(log_dir, 'samples')}")

    # Finish wandb run
    if is_main_process and wandb_run is not None:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: Failed to finish wandb run: {e}")

    # Final barrier before cleanup
    if is_distributed:
        dist.barrier()
        cleanup_distributed(is_distributed)



def main():
    parser = argparse.ArgumentParser(description='Train diffusion models')
    parser.add_argument('--method', type=str, required=True,
                       choices=['ddpm', 'ddpmx0', 'flow_matching', 'mean_flow', 'progressive_distillation'],
                       help='Method to train')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (e.g., configs/ddpm.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to a pretrained FM checkpoint for MeanFlow weight init')
    parser.add_argument('--overfit-single-batch', action='store_true',
                       help='DEBUG MODE: Train on a single batch repeatedly to verify model can overfit')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with resume path if specified
    if args.resume:
        config['checkpoint']['resume'] = args.resume

    # Train
    train(
        method_name=args.method,
        config=config,
        resume_path=args.resume,
        pretrained_path=args.pretrained,
        overfit_single_batch=args.overfit_single_batch,
    )


if __name__ == '__main__':
    main()
