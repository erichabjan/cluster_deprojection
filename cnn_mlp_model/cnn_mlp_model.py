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
    Only smooth discrete channels, identity-initialized.
    Expects channels-last images: (B, H, W, C).
    """
    kernel_size: int = 5
    discrete_channel_indices: Sequence[int] = (1, 2)

    @staticmethod
    def _identity_kernel_init(kernel_size: int):
        def init(key, shape, dtype=jnp.float32):
            kh, kw, in_ch, out_ch = shape
            k = jnp.zeros(shape, dtype=dtype)
            cy = kh // 2
            cx = kw // 2
            k = k.at[cy, cx, 0, 0].set(1.0)
            return k
        return init

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        k = self.kernel_size
        if k % 2 != 1:
            raise ValueError("kernel_size should be odd for identity init.")

        outs = [x[..., i:i+1] for i in range(C)]
        for c in self.discrete_channel_indices:
            xc = x[..., c:c+1]
            yc = nn.Conv(
                features=1,
                kernel_size=(k, k),
                padding="SAME",
                use_bias=True,
                kernel_init=self._identity_kernel_init(k),
                bias_init=nn.initializers.zeros,
                name=f"smooth_identity_c{c}",
            )(xc)
            outs[c] = yc
        return jnp.concatenate(outs, axis=-1)


class ResidualBlock(nn.Module):
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
    channels: Sequence[int] = (32, 64, 128)
    blocks_per_stage: int = 2
    dropout_rate: float = 0.0
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, deterministic: bool):
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
    B, H, W, C = feature_map.shape
    _, N, two = pixel_coords.shape
    assert two == 2

    x = jnp.clip(pixel_coords[..., 0], 0.0, W - 1.0)
    y = jnp.clip(pixel_coords[..., 1], 0.0, H - 1.0)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y1 = jnp.clip(y0 + 1, 0, H - 1)

    wx = x - x0.astype(x.dtype)
    wy = y - y0.astype(y.dtype)

    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w10 = wx * (1.0 - wy)
    w11 = wx * wy

    b_idx = jnp.arange(B, dtype=jnp.int32)[:, None]
    b_idx = jnp.broadcast_to(b_idx, (B, N))

    f00 = feature_map[b_idx, y0, x0]
    f01 = feature_map[b_idx, y1, x0]
    f10 = feature_map[b_idx, y0, x1]
    f11 = feature_map[b_idx, y1, x1]

    sampled = (
        w00[..., None] * f00
        + w01[..., None] * f01
        + w10[..., None] * f10
        + w11[..., None] * f11
    )
    return sampled


def bilinear_sample_feature_map_window(feature_map: jnp.ndarray, pixel_coords: jnp.ndarray, window_size: int) -> jnp.ndarray:
    if window_size % 2 != 1 or window_size < 1:
        raise ValueError(f"window_size must be an odd positive int, got {window_size}")

    B, H, W, C = feature_map.shape
    _, N, _ = pixel_coords.shape

    r = window_size // 2
    dx = jnp.arange(-r, r + 1, dtype=pixel_coords.dtype)
    dy = jnp.arange(-r, r + 1, dtype=pixel_coords.dtype)
    off_y, off_x = jnp.meshgrid(dy, dx, indexing="ij")
    offsets = jnp.stack([off_x.reshape(-1), off_y.reshape(-1)], axis=-1)  # (P,2)
    P = offsets.shape[0]

    coords = pixel_coords[:, :, None, :] + offsets[None, None, :, :]      # (B,N,P,2)
    coords = coords.reshape(B, N * P, 2)                                  # (B,N*P,2)

    sampled = bilinear_sample_feature_map(feature_map, coords)            # (B,N*P,C)
    sampled = sampled.reshape(B, N, P * C)                                # (B,N,P*C)
    return sampled


@dataclass
class ModelConfig:
    smoother_kernel: int = 5

    fmap_channels: Sequence[int] = (32, 64, 128)
    fmap_blocks_per_stage: int = 2
    cnn_dropout: float = 0.0

    sampler_window: int = 5  # 5x5

    # NEW: global context
    global_dim: int = 128
    global_use_std: bool = True  # mean+max (+std)

    mlp_hidden: Sequence[int] = (128, 128)
    head_hidden: Sequence[int] = (256, 256)
    dropout: float = 0.0
    output_dim: int = 1


class CNNMLPModel(nn.Module):
    cfg: ModelConfig

    def setup(self):
        self.smoother = DiscreteMapSmoother(
            kernel_size=self.cfg.smoother_kernel,
            discrete_channel_indices=(1, 2),
        )
        self.encoder = ResidualFeatureMapEncoder(
            channels=self.cfg.fmap_channels,
            blocks_per_stage=self.cfg.fmap_blocks_per_stage,
            dropout_rate=self.cfg.cnn_dropout,
        )
        self.global_proj = nn.Dense(self.cfg.global_dim)

        self.query_mlp = MLP(
            [*self.cfg.mlp_hidden, self.cfg.mlp_hidden[-1]],
            dropout_rate=self.cfg.dropout,
        )
        self.head_mlp = MLP(
            [*self.cfg.head_hidden, self.cfg.output_dim],
            dropout_rate=self.cfg.dropout,
        )

    def _global_context(self, fmap: jnp.ndarray, N: int) -> jnp.ndarray:
        """
        fmap: (B,H,W,Cf) -> returns (B,N,global_dim)
        """
        g_mean = jnp.mean(fmap, axis=(1, 2))  # (B,Cf)
        g_max = jnp.max(fmap, axis=(1, 2))    # (B,Cf)

        if self.cfg.global_use_std:
            # variance over spatial dims
            mu = g_mean[:, None, None, :]
            g_std = jnp.sqrt(jnp.mean((fmap - mu) ** 2, axis=(1, 2)) + 1e-6)  # (B,Cf)
            g = jnp.concatenate([g_mean, g_max, g_std], axis=-1)              # (B,3Cf)
        else:
            g = jnp.concatenate([g_mean, g_max], axis=-1)                     # (B,2Cf)

        g = self.global_proj(g)  # (B,global_dim)
        g = g[:, None, :]        # (B,1,global_dim)
        g = jnp.broadcast_to(g, (g.shape[0], N, g.shape[-1]))  # (B,N,global_dim)
        return g

    def __call__(self, images, mlp_in, gal_pixel_coords, deterministic: bool):
        # images: (B,H,W,C), mlp_in: (B,N,F), gal_pixel_coords: (B,N,2)
        B, N, _ = mlp_in.shape

        smoothed = self.smoother(images)
        fmap = self.encoder(smoothed, deterministic=deterministic)  # (B,H,W,Cf)

        sampled_feat = bilinear_sample_feature_map_window(
            fmap, gal_pixel_coords, window_size=self.cfg.sampler_window
        )  # (B,N,(ws^2)*Cf)

        q = self.query_mlp(mlp_in, deterministic=deterministic)  # (B,N,Dq)

        g = self._global_context(fmap, N=N)  # (B,N,global_dim)

        fused = jnp.concatenate([sampled_feat, q, g], axis=-1)
        out = self.head_mlp(fused, deterministic=deterministic)  # (B,N,output_dim)
        return out