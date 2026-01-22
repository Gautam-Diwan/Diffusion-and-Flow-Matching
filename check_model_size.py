#!/usr/bin/env python3
"""
Script to check UNet model size and architecture details
"""
import torch
import yaml
from src.models.unet import UNet, create_model_from_config

def format_size(bytes_size):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def analyze_model(model):
    """Analyze model size and parameters"""
    total_params = 0
    total_trainable = 0
    total_non_trainable = 0
    
    print("\n" + "="*70)
    print("UNet Model Analysis")
    print("="*70)
    
    # Count parameters by layer
    print("\nLayer-wise breakdown:")
    print("-" * 70)
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Embedding)):
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:50s}: {params:>12,d} params")
    
    # Total summary
    print("\n" + "-" * 70)
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            total_trainable += num_params
        else:
            total_non_trainable += num_params
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {total_trainable:,}")
    print(f"Non-trainable Parameters: {total_non_trainable:,}")
    
    # Memory estimate (for fp32)
    memory_fp32 = total_params * 4  # 4 bytes per float32
    # Memory estimate (for fp16)
    memory_fp16 = total_params * 2  # 2 bytes per float16
    
    print(f"\nMemory footprint (weights only):")
    print(f"  FP32: {format_size(memory_fp32)}")
    print(f"  FP16: {format_size(memory_fp16)}")
    
    # Forward pass memory estimate
    batch_size = 1
    input_size = (batch_size, 3, 64, 64)
    activations_fp32 = batch_size * 3 * 64 * 64 * 4  # rough estimate
    total_forward_fp32 = memory_fp32 + activations_fp32
    
    print(f"\nEstimated memory for forward pass (batch_size={batch_size}, FP32):")
    print(f"  Weights: {format_size(memory_fp32)}")
    print(f"  Activations (rough): {format_size(activations_fp32)}")
    print(f"  Total: {format_size(total_forward_fp32)}")
    
    return total_params

def main():
    # Load config
    with open("configs/ddpm.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("\nConfig details:")
    print(f"  Base channels: {config['model']['base_channels']}")
    print(f"  Channel multipliers: {config['model']['channel_mult']}")
    print(f"  Num res blocks: {config['model']['num_res_blocks']}")
    print(f"  Attention resolutions: {config['model']['attention_resolutions']}")
    print(f"  Num heads: {config['model']['num_heads']}")
    print(f"  Input size: {config['data']['image_size']}x{config['data']['image_size']}")
    
    # Create model from config
    model = create_model_from_config(config)
    model.eval()
    
    # Analyze
    total_params = analyze_model(model)
    
    # Test forward pass
    print("\n" + "="*70)
    print("Forward Pass Test")
    print("="*70)
    with torch.no_grad():
        x = torch.randn(4, 3, 64, 64)
        t = torch.randint(0, 1000, (4,))
        out = model(x, t)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        print(f"Forward pass: ✓ Success")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
