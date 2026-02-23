"""
Sampling Script for DDPM (Denoising Diffusion Probabilistic Models)

Generate samples from a trained model. By default, saves individual images to avoid
memory issues with large sample counts. Use --grid to generate a single grid image.
"""

import os
import sys
import argparse
import math  # Added math for grid calculation
from datetime import datetime

import yaml
import torch
from tqdm import tqdm
import json

from src.models import create_model_from_config, create_meanflow_model_from_config
from src.data import save_image
from src.methods import DDPM
from src.methods.flow_matching import FlowMatching
from src.methods.mean_flow import MeanFlow
from src.utils import EMA


def load_checkpoint(checkpoint_path: str, device: torch.device, method: str = "ddpm"):
    """Load checkpoint and return model, config, and EMA."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Create model (MeanFlow uses UNetMeanFlow, others use UNet)
    if method == 'mean_flow':
        model = create_meanflow_model_from_config(config).to(device)
    else:
        model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint['model'])
    
    # Create EMA only if checkpoint contains it.
    ema = None
    if 'ema' in checkpoint:
        ema = EMA(model, decay=config['training']['ema_decay'])
        ema.load_state_dict(checkpoint['ema'])
    
    return model, config, ema


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    num_samples: int,
    nrow: int = None,
) -> None:
    """
    Save generated samples as images. Unnormalizes from [-1, 1] to [0, 1].

    Args:
        samples: Generated samples tensor with shape (num_samples, C, H, W).
        save_path: File path to save the image grid.
        num_samples: Number of samples.
        nrow: Images per row for grids. If None, calculated automatically.
    """
    # 1. Unnormalize: Convert from [-1, 1] range back to [0, 1]
    # This fixes the "Dark Image" issue
    samples = (samples + 1.0) / 2.0

    # 2. Clamp to ensure numerical stability (keep within valid image range)
    samples = torch.clamp(samples, 0.0, 1.0)

    # 3. Calculate grid layout if not provided
    if nrow is None:
        if num_samples <= 1:
            nrow = 1
        else:
            nrow = int(math.ceil(math.sqrt(num_samples)))
    
    # 4. Save using the helper from src.data
    save_image(samples, save_path, nrow=nrow)


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--method', type=str, required=True,
                       choices=['ddpm', 'flow_matching', 'mean_flow', 'progressive_distillation'],
                       help='Method used for training')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='samples',
                       help='Directory to save individual images (default: samples)')
    parser.add_argument('--grid', action='store_true',
                       help='Save as grid image instead of individual images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for grid (only used with --grid, default: samples_<timestamp>.png)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Sampling arguments
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of sampling steps (default: from config)')
    parser.add_argument('--sampler', type=str, default=None,
                   choices=['ddpm', 'ddim', 'dpm_solver', 'euler', 'heun'],
                   help='Sampling method (default: from config/method)')
    parser.add_argument('--order', type=int, default=None,
	                   help='DPM-Solver order (1, 2, or 3)')
    parser.add_argument('--dpm_method', type=str, default=None,
	                   choices=['singlestep', 'multistep'],
	                   help='DPM-Solver method')
    parser.add_argument('--skip_type', type=str, default=None,
	                   choices=['time_uniform', 'logSNR'],
	                   help='DPM-Solver timestep schedule')
    
    # Other options
    parser.add_argument('--no_ema', action='store_true',
                       help='Use training weights instead of EMA weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, ema = load_checkpoint(args.checkpoint, device, method=args.method)

    # Create method
    if args.method == 'ddpm':
        method = DDPM.from_config(model, config, device)
    elif args.method == 'flow_matching':
        method = FlowMatching.from_config(model, config, device)
    elif args.method == 'progressive_distillation':
        # Distilled student checkpoints are sampled with FlowMatching samplers.
        method = FlowMatching.from_config(model, config, device)
    elif args.method == 'mean_flow':
        method = MeanFlow.from_config(model, config, device)
    else:
        raise ValueError(
            f"Unknown method: {args.method}. "
            "Supported: ddpm, flow_matching, mean_flow, progressive_distillation."
        )
    
    # Apply EMA weights
    if ema is not None and not args.no_ema:
        print("Using EMA weights")
        ema.apply_shadow()
    elif ema is None:
        print("No EMA weights found in checkpoint, using training weights")
    else:
        print("Using training weights (no EMA)")
    
    method.eval_mode()
    
    # Image shape
    data_config = config['data']
    image_shape = (data_config['channels'], data_config['image_size'], data_config['image_size'])
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")

    all_samples = []
    remaining = args.num_samples
    sample_idx = 0

    # Create output directory if saving individual images
    if not args.grid:
        os.makedirs(args.output_dir, exist_ok=True)

    all_samples = []
    all_metrics = []

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating samples")
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)

            num_steps = args.num_steps or config['sampling']['num_steps']
            default_sampler = (
                'meanflow'
                if args.method == 'mean_flow'
                else ('dpm_solver' if args.method == 'progressive_distillation' else ('euler' if args.method == 'flow_matching' else 'ddpm'))
            )
            sampler = args.sampler or config['sampling'].get('sampler', default_sampler)

            print(f"Sampling batch of size {batch_size} with sampler '{sampler}' and num steps: {num_steps}")

            # Build kwargs for DPM-Solver (ignored by MeanFlow)
            sampling_kwargs = {'num_steps': num_steps}
            if sampler == 'dpm_solver':
                sampling_kwargs['sampler'] = sampler
                dpm_config = config['sampling'].get('dpm_solver', {})
                sampling_kwargs['order'] = args.order or dpm_config.get('order', 2)
                sampling_kwargs['method'] = args.dpm_method or dpm_config.get('method', 'multistep')
                sampling_kwargs['skip_type'] = args.skip_type or dpm_config.get('skip_type', 'time_uniform')
            else:
                sampling_kwargs['sampler'] = sampler

            samples, metrics = method.sample(
                batch_size=batch_size,
                image_shape=image_shape,
                **sampling_kwargs
            )
            
            all_metrics.append(metrics)

            # Save individual images immediately or collect for grid
            if args.grid:
                all_samples.append(samples)
            else:
                for i in range(samples.shape[0]):
                    img_path = os.path.join(args.output_dir, f"{sample_idx:06d}.png")
                    # Pass single image (unsqueeze to make it 4D: 1, C, H, W)
                    save_samples(samples[i].unsqueeze(0), img_path, 1)
                    sample_idx += 1

            remaining -= batch_size
            pbar.update(batch_size)

        pbar.close()

    # Aggregate metrics
    total_nfe = sum(m['nfe'] for m in all_metrics)
    total_time = sum(m['wall_clock_time'] for m in all_metrics)
    avg_nfe_per_sample = total_nfe / args.num_samples
    avg_time_per_sample = total_time / args.num_samples
    
    summary_metrics = {
        'total_samples': args.num_samples,
        'total_nfe': total_nfe,
        'total_wall_clock_time': total_time,
        'avg_nfe_per_sample': avg_nfe_per_sample,
        'avg_time_per_sample': avg_time_per_sample,
        'sampler': sampler,
        'num_steps': num_steps,
    }

    # Save samples
    if args.grid:
        # Concatenate all samples for grid
        all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]

        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"samples_{timestamp}.png"

        save_samples(all_samples, args.output, num_samples=args.num_samples, nrow=8)
        print(f"Saved grid to {args.output}")
    else:
        print(f"Saved {args.num_samples} individual images to {args.output_dir}")

    # Restore EMA if applied
    if ema is not None and not args.no_ema:
        ema.restore()

    print(f"\nSampling Metrics:")
    print(f"  Total NFE: {total_nfe}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg NFE/sample: {avg_nfe_per_sample:.1f}")
    print(f"  Avg Time/sample: {avg_time_per_sample:.3f}s")


if __name__ == '__main__':
    main()