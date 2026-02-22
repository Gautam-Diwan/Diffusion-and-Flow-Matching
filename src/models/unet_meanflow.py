"""
U-Net for MeanFlow: dual-time conditioning (t, t-r).

Same architecture as UNet but time embedding is built from (t, t-r) using
sinusoidal embeddings, for predicting average velocity u_θ(z, r, t).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .blocks import (
    SinusoidalPositionalEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNetMeanFlow(nn.Module):
    """
    U-Net with (t, t-r) time conditioning for MeanFlow.
    Forward: (x, r, t) -> u, with r, t continuous in [0, 1].
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

        # Dual-time embedding: (t, t-r) -> concat sinusoidal -> project to time_embed_dim
        time_embed_dim = base_channels * 4
        emb_dim = base_channels // 2  # each of t and (t-r) gets half
        self.sinusoidal_t = SinusoidalPositionalEmbedding(emb_dim)
        self.sinusoidal_dt = SinusoidalPositionalEmbedding(emb_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

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

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            skip_ch = out_ch

            for i in range(num_res_blocks + 1):
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

    def forward(self, x: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, height, width)
            r: (batch_size,) continuous in [0, 1]
            t: (batch_size,) continuous in [0, 1], t >= r

        Returns:
            (batch_size, out_channels, height, width)
        """
        # r, t may be scalars or (B,); ensure (B,) for embedding
        if r.dim() == 0:
            r = r.expand(x.shape[0])
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        dt = (t - r).clamp(min=1e-5)  # avoid div by zero; dt=0 handled by clamp

        t_emb_t = self.sinusoidal_t(t)
        t_emb_dt = self.sinusoidal_dt(dt)
        t_emb = torch.cat([t_emb_t, t_emb_dt], dim=-1)
        t_emb = self.time_embed(t_emb)

        # Initial convolution
        h = self.input_conv(x)
        hs = []

        # Encoder
        encoder_block_idx = 0
        down_idx = 0

        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                block = self.encoder_blocks[encoder_block_idx]
                h = block(h, t_emb)
                encoder_block_idx += 1

                if 64 // (2**level) in self.attention_resolutions:
                    block = self.encoder_blocks[encoder_block_idx]
                    h = block(h)
                    encoder_block_idx += 1

            hs.append(h)

            if level != len(self.channel_mult) - 1:
                h = self.encoder_downsample[down_idx](h)
                down_idx += 1

        # Middle
        for block in self.middle_blocks:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:
                h = block(h)

        # Decoder
        decoder_block_idx = 0
        up_idx = 0

        for level, mult in reversed(list(enumerate(self.channel_mult))):
            skip = hs.pop() if len(hs) > 0 else None

            for i in range(self.num_res_blocks + 1):
                if i == 0 and skip is not None:
                    h = torch.cat([h, skip], dim=1)

                block = self.decoder_blocks[decoder_block_idx]
                h = block(h, t_emb)
                decoder_block_idx += 1

                if 64 // (2**level) in self.attention_resolutions:
                    block = self.decoder_blocks[decoder_block_idx]
                    h = block(h)
                    decoder_block_idx += 1

            if level != 0:
                h = self.decoder_upsample[up_idx](h)
                up_idx += 1

        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)

        return h


def create_meanflow_model_from_config(config: dict) -> UNetMeanFlow:
    """Factory: build UNetMeanFlow from config (same keys as create_model_from_config)."""
    model_config = config["model"]
    data_config = config["data"]

    return UNetMeanFlow(
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
