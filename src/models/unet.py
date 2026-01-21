"""
U-Net Architecture for Diffusion Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks

    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3,
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        # Time embedding
        time_embed_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.time_pos_emb = nn.Embedding(10000, base_channels)

        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        input_channels = base_channels

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock(
                        in_channels=input_channels,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                input_channels = out_ch

                if 64 // (2**level) in attention_resolutions:
                    self.encoder_blocks.append(AttentionBlock(channels=out_ch, num_heads=num_heads))

            if level != len(channel_mult) - 1:
                self.encoder_downsample.append(Downsample(out_ch))

        # Middle
        self.middle_blocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=input_channels,
                    out_channels=input_channels,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                AttentionBlock(channels=input_channels, num_heads=num_heads),
                ResBlock(
                    in_channels=input_channels,
                    out_channels=input_channels,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            ]
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()

        # For decoder, we go through levels in reverse
        # We need to know what the skip connection channels are
        # Skips come from the end of each encoder level, so they have channels = base_channels * mult
        
        # input_channels at this point is the output of the middle blocks = base_channels * channel_mult[-1]
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            # Skip connection has out_ch channels (from the encoder at this level)
            skip_ch = out_ch
            
            for i in range(num_res_blocks + 1):
                # First block gets concatenation of skip + current
                if i == 0:
                    decoder_in_ch = input_channels + skip_ch
                else:
                    decoder_in_ch = out_ch
                
                self.decoder_blocks.append(
                    ResBlock(
                        in_channels=decoder_in_ch,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                
                if 64 // (2**level) in attention_resolutions:
                    self.decoder_blocks.append(AttentionBlock(channels=out_ch, num_heads=num_heads))
            
            input_channels = out_ch
            if level != 0:
                self.decoder_upsample.append(Upsample(out_ch))

        # Output
        self.final_norm = GroupNorm32(32, base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch_size, in_channels, height, width)
            t: Timestep (batch_size,) in range [0, num_timesteps)

        Returns:
            Output (batch_size, out_channels, height, width)
        """
        # Time embedding
        t_emb = self.time_pos_emb(t)
        t_emb = self.time_embed(t_emb)

        # Initial convolution
        h = self.input_conv(x)
        hs = []  # Store skip connections at each resolution

        # Encoder with downsampling
        encoder_block_idx = 0
        down_idx = 0
        
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                # ResBlock
                block = self.encoder_blocks[encoder_block_idx]
                h = block(h, t_emb)
                encoder_block_idx += 1
                
                # AttentionBlock (if applicable)
                if 64 // (2**level) in self.attention_resolutions:
                    block = self.encoder_blocks[encoder_block_idx]
                    h = block(h)
                    encoder_block_idx += 1
            
            # Store skip connection BEFORE downsampling
            hs.append(h)
            
            # Downsample at the end of each level (except last)
            if level != len(self.channel_mult) - 1:
                h = self.encoder_downsample[down_idx](h)
                down_idx += 1

        # Middle blocks
        for block in self.middle_blocks:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:  # AttentionBlock
                h = block(h)

        # Decoder with upsampling
        decoder_block_idx = 0
        up_idx = 0
        
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            # Get skip connection for this level
            skip = hs.pop() if len(hs) > 0 else None
            
            for i in range(self.num_res_blocks + 1):
                # Concatenate skip connection only on the first block of each level
                if i == 0 and skip is not None:
                    h = torch.cat([h, skip], dim=1)
                
                # ResBlock
                block = self.decoder_blocks[decoder_block_idx]
                h = block(h, t_emb)
                decoder_block_idx += 1
                
                # AttentionBlock (if applicable)
                if 64 // (2**level) in self.attention_resolutions:
                    block = self.decoder_blocks[decoder_block_idx]
                    h = block(h)
                    decoder_block_idx += 1
            
            # Upsample at the end of each level (except first)
            if level != 0:
                h = self.decoder_upsample[up_idx](h)
                up_idx += 1

        # Output
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)

        return h


def create_model_from_config(config: dict) -> UNet:
    """Factory function to create UNet from config."""
    model_config = config["model"]
    data_config = config["data"]

    return UNet(
        in_channels=data_config["channels"],
        out_channels=data_config["channels"],
        base_channels=model_config["base_channels"],
        channel_mult=tuple(model_config["channel_mult"]),
        num_res_blocks=model_config["num_res_blocks"],
        attention_resolutions=model_config["attention_resolutions"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        use_scale_shift_norm=model_config["use_scale_shift_norm"],
    )


if __name__ == "__main__":
    print("Testing UNet...")

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e6:.2f}M")

    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.randint(0, 1000, (batch_size,))

    with torch.no_grad():
        out = model(x, t)

    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print("Forward pass successful!")
