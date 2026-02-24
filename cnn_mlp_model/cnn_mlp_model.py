from dataclasses import dataclass
from typing import Sequence, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    feature_sizes: Sequence[int]
    activation: Callable = nn.gelu
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        for feat in self.feature_sizes[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.feature_sizes[-1])(x)
        return x


class DiscreteMapSmoother(nn.Module):
    """
    Channel-wise learnable smoothing on image inputs.
    Expects channels-last images: (B, H, W, C)
    """
    kernel_size: int = 5

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        outs = []
        for c in range(C):
            xc = x[..., c:c+1]
            yc = nn.Conv(
                features=1,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding="SAME",
                use_bias=True,
                kernel_init=nn.initializers.normal(stddev=0.02),
                bias_init=nn.initializers.zeros,
                name=f"smooth_conv_c{c}",
            )(xc)
            outs.append(yc)
        return jnp.concatenate(outs, axis=-1)


class ResidualBlock(nn.Module):
    """
    Same-resolution residual block.
    """
    channels: int
    dropout_rate: float = 0.0
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, deterministic: bool):
        in_ch = x.shape[-1]
        residual = x

        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = self.activation(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)

        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)

        if in_ch != self.channels:
            residual = nn.Conv(self.channels, kernel_size=(1, 1), padding="SAME")(residual)

        x = x + residual
        x = self.activation(x)
        return x


class ResidualFeatureMapEncoder(nn.Module):
    """
    CNN encoder that preserves spatial resolution and returns a feature map.
    No pooling / no striding.
    """
    channels: Sequence[int] = (32, 64, 128)
    blocks_per_stage: int = 2
    dropout_rate: float = 0.0
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, deterministic: bool):
        """
        x: (B, H, W, C_in)
        returns: (B, H, W, C_out)
        """
        x = nn.Conv(self.channels[0], kernel_size=(3, 3), padding="SAME")(x)
        x = self.activation(x)

        for si, ch in enumerate(self.channels):
            for bi in range(self.blocks_per_stage):
                x = ResidualBlock(
                    channels=ch,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    name=f"res_stage{si}_block{bi}",
                )(x, deterministic=deterministic)

        return x


def bilinear_sample_feature_map(feature_map: jnp.ndarray, pixel_coords: jnp.ndarray) -> jnp.ndarray:
    """
    Bilinear sampler for per-galaxy feature queries.

    Args:
        feature_map: (B, H, W, C)
        pixel_coords: (B, N, 2) continuous coords in image pixel space
                     [..., 0] = x_pix in [0, W-1]
                     [..., 1] = y_pix in [0, H-1]

    Returns:
        sampled: (B, N, C)
    """
    B, H, W, C = feature_map.shape
    _, N, two = pixel_coords.shape
    assert two == 2, f"pixel_coords must have shape (B, N, 2), got {pixel_coords.shape}"

    x = pixel_coords[..., 0]
    y = pixel_coords[..., 1]

    # Clamp to valid range
    x = jnp.clip(x, 0.0, W - 1.0)
    y = jnp.clip(y, 0.0, H - 1.0)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y1 = jnp.clip(y0 + 1, 0, H - 1)

    # Fractional part
    wx = x - x0.astype(x.dtype)
    wy = y - y0.astype(y.dtype)

    # Bilinear weights
    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w10 = wx * (1.0 - wy)
    w11 = wx * wy

    # Batch indices for gather
    b_idx = jnp.arange(B, dtype=jnp.int32)[:, None]
    b_idx = jnp.broadcast_to(b_idx, (B, N))

    # Gather corner features: each -> (B, N, C)
    f00 = feature_map[b_idx, y0, x0]
    f01 = feature_map[b_idx, y1, x0]
    f10 = feature_map[b_idx, y0, x1]
    f11 = feature_map[b_idx, y1, x1]

    # Weighted sum (expand weights to feature dim)
    sampled = (
        w00[..., None] * f00
        + w01[..., None] * f01
        + w10[..., None] * f10
        + w11[..., None] * f11
    )
    return sampled


@dataclass
class ModelConfig:
    smoother_kernel: int = 5

    # New residual feature-map encoder config
    fmap_channels: Sequence[int] = (32, 64, 128)
    fmap_blocks_per_stage: int = 2
    cnn_dropout: float = 0.0

    # MLP branches
    mlp_hidden: Sequence[int] = (128, 128)
    head_hidden: Sequence[int] = (256, 256)
    dropout: float = 0.0
    output_dim: int = 3


class CNNMLPModel(nn.Module):
    """
    Model with:
      - CNN residual feature-map encoder (spatially preserved)
      - bilinear sampler using per-galaxy pixel coords
      - per-galaxy MLP branch
      - fusion head MLP

    Expected inputs:
      images:           (B, H, W, C)  [channels-last]
      mlp_in:           (B, N, Fm)    e.g. Fm=4
      gal_pixel_coords: (B, N, 2)     [x_pix, y_pix] in image pixel coordinates
    """
    cfg: ModelConfig

    @nn.compact
    def __call__(self, images, mlp_in, gal_pixel_coords, deterministic: bool):
        """
        Args:
            images: (B, H, W, C)  -- if your dataloader gives (B, C, H, W), transpose before calling
            mlp_in: (B, N, 4)
            gal_pixel_coords: (B, N, 2), columns=[x_pix, y_pix]
            deterministic: bool

        Returns:
            out: (B, N, output_dim)
        """
        # --- Input sanity ---
        if images.ndim != 4:
            raise ValueError(f"images must be rank-4 (B,H,W,C), got shape {images.shape}")
        if mlp_in.ndim != 3:
            raise ValueError(f"mlp_in must be rank-3 (B,N,F), got shape {mlp_in.shape}")
        if gal_pixel_coords.ndim != 3 or gal_pixel_coords.shape[-1] != 2:
            raise ValueError(
                f"gal_pixel_coords must have shape (B,N,2), got {gal_pixel_coords.shape}"
            )

        B_img = images.shape[0]
        B_mlp, N, _ = mlp_in.shape
        B_pix, N_pix, _ = gal_pixel_coords.shape
        if not (B_img == B_mlp == B_pix):
            raise ValueError(f"Batch mismatch: images={B_img}, mlp_in={B_mlp}, pixel_coords={B_pix}")
        if N != N_pix:
            raise ValueError(f"Galaxy count mismatch: mlp_in N={N}, pixel_coords N={N_pix}")

        smoothed_images = DiscreteMapSmoother(kernel_size=self.cfg.smoother_kernel)(images)

        feature_map = ResidualFeatureMapEncoder(channels=self.cfg.fmap_channels, blocks_per_stage=self.cfg.fmap_blocks_per_stage,
                                                dropout_rate=self.cfg.cnn_dropout,)(smoothed_images, deterministic=deterministic)

        sampled_feat = bilinear_sample_feature_map(feature_map, gal_pixel_coords)

        q = MLP([*self.cfg.mlp_hidden, self.cfg.mlp_hidden[-1]], dropout_rate=self.cfg.dropout,)(mlp_in, deterministic=deterministic)

        fused = jnp.concatenate([sampled_feat, q], axis=-1)

        out = MLP([*self.cfg.head_hidden, self.cfg.output_dim], dropout_rate=self.cfg.dropout,)(fused, deterministic=deterministic)

        return out