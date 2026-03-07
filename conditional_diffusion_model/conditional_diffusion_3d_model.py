from dataclasses import dataclass
from typing import Sequence, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn


def sinusoidal_time_embedding(timesteps: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    timesteps: (B,) int32 or float32
    returns: (B, dim)
    """
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(0, half) / max(half - 1, 1))
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class MLP(nn.Module):
    feature_sizes: Sequence[int]
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        for feat in self.feature_sizes[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
        x = nn.Dense(self.feature_sizes[-1])(x)
        return x


class ResBlock3D(nn.Module):
    out_ch: int
    cond_dim: int
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, cond_vec):
        """
        x: (B, Z, Y, X, C)
        cond_vec: (B, cond_dim)
        """
        in_ch = x.shape[-1]
        residual = x

        h = nn.Conv(self.out_ch, kernel_size=(3, 3, 3), padding="SAME")(x)
        h = nn.GroupNorm(num_groups=min(8, self.out_ch))(h)

        cond = nn.Dense(2 * self.out_ch)(cond_vec)  # (B, 2C)
        scale, shift = jnp.split(cond, 2, axis=-1)
        scale = scale[:, None, None, None, :]
        shift = shift[:, None, None, None, :]
        h = h * (1.0 + scale) + shift

        h = self.activation(h)
        h = nn.Conv(self.out_ch, kernel_size=(3, 3, 3), padding="SAME")(h)
        h = nn.GroupNorm(num_groups=min(8, self.out_ch))(h)
        h = self.activation(h)

        if in_ch != self.out_ch:
            residual = nn.Conv(self.out_ch, kernel_size=(1, 1, 1), padding="SAME")(residual)

        return h + residual


class Downsample3D(nn.Module):
    out_ch: int

    @nn.compact
    def __call__(self, x):
        return nn.Conv(
            self.out_ch,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding="SAME",
        )(x)


class Upsample3D(nn.Module):
    out_ch: int

    @nn.compact
    def __call__(self, x):
        B, Z, Y, X, C = x.shape
        x = jax.image.resize(x, shape=(B, Z * 2, Y * 2, X * 2, C), method="nearest")
        x = nn.Conv(self.out_ch, kernel_size=(3, 3, 3), padding="SAME")(x)
        return x


class Conditioning2DEncoder(nn.Module):
    channels: Sequence[int] = (32, 64, 128)
    emb_dim: int = 128
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, images):
        """
        images: (B,H,W,C)
        """
        x = images
        for ch in self.channels:
            x = nn.Conv(ch, kernel_size=(3, 3), padding="SAME")(x)
            x = self.activation(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        x_mean = jnp.mean(x, axis=(1, 2))
        x_max = jnp.max(x, axis=(1, 2))
        x = jnp.concatenate([x_mean, x_max], axis=-1)
        x = MLP([self.emb_dim, self.emb_dim])(x)
        return x


@dataclass
class DiffusionModelConfig:
    base_channels: int = 32
    channel_mults: Sequence[int] = (1, 2, 4)
    time_emb_dim: int = 128
    cond_emb_dim: int = 128
    out_channels: int = 1


class ConditionalUNet3D(nn.Module):
    cfg: DiffusionModelConfig

    @nn.compact
    def __call__(self, noisy_cube, timesteps, cond_images):
        """
        noisy_cube: (B,Z,Y,X,1)
        timesteps: (B,)
        cond_images: (B,H,W,3)
        returns predicted noise with same shape as noisy_cube
        """
        time_emb = sinusoidal_time_embedding(timesteps, self.cfg.time_emb_dim)
        time_emb = MLP([self.cfg.time_emb_dim, self.cfg.time_emb_dim])(time_emb)

        cond_emb = Conditioning2DEncoder(
            emb_dim=self.cfg.cond_emb_dim
        )(cond_images)

        cond_vec = jnp.concatenate([time_emb, cond_emb], axis=-1)

        chs = [self.cfg.base_channels * m for m in self.cfg.channel_mults]

        x = nn.Conv(chs[0], kernel_size=(3, 3, 3), padding="SAME")(noisy_cube)

        skips = []
        for ch in chs[:-1]:
            x = ResBlock3D(ch, cond_dim=cond_vec.shape[-1])(x, cond_vec)
            x = ResBlock3D(ch, cond_dim=cond_vec.shape[-1])(x, cond_vec)
            skips.append(x)
            x = Downsample3D(ch)(x)

        x = ResBlock3D(chs[-1], cond_dim=cond_vec.shape[-1])(x, cond_vec)
        x = ResBlock3D(chs[-1], cond_dim=cond_vec.shape[-1])(x, cond_vec)

        for ch, skip in zip(reversed(chs[:-1]), reversed(skips)):
            x = Upsample3D(ch)(x)

            # crop if needed due to odd shapes
            min_z = min(x.shape[1], skip.shape[1])
            min_y = min(x.shape[2], skip.shape[2])
            min_x = min(x.shape[3], skip.shape[3])

            x = x[:, :min_z, :min_y, :min_x, :]
            skip = skip[:, :min_z, :min_y, :min_x, :]

            x = jnp.concatenate([x, skip], axis=-1)
            x = ResBlock3D(ch, cond_dim=cond_vec.shape[-1])(x, cond_vec)
            x = ResBlock3D(ch, cond_dim=cond_vec.shape[-1])(x, cond_vec)

        x = nn.GroupNorm(num_groups=min(8, x.shape[-1]))(x)
        x = nn.gelu(x)
        x = nn.Conv(self.cfg.out_channels, kernel_size=(3, 3, 3), padding="SAME")(x)
        return x