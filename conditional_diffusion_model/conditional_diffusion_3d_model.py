from dataclasses import dataclass
from typing import Sequence, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn


def sinusoidal_time_embedding(timesteps: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    timesteps: (B,)
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
    activate_last: bool = False

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.feature_sizes):
            x = nn.Dense(feat)(x)
            if i < len(self.feature_sizes) - 1 or self.activate_last:
                x = self.activation(x)
        return x


class ResBlock3D(nn.Module):
    out_ch: int
    time_dim: int
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, time_emb, cond_xyz):
        """
        x:        (B, Z, Y, X, C_in)
        time_emb: (B, time_dim)
        cond_xyz: (B, Z, Y, X, C_cond)
        """
        in_ch = x.shape[-1]
        residual = x

        h = nn.Conv(self.out_ch, kernel_size=(3, 3, 3), padding="SAME")(x)
        h = nn.GroupNorm(num_groups=min(8, self.out_ch))(h)

        # Global time conditioning
        t_cond = nn.Dense(2 * self.out_ch)(time_emb)
        t_scale, t_shift = jnp.split(t_cond, 2, axis=-1)
        t_scale = t_scale[:, None, None, None, :]
        t_shift = t_shift[:, None, None, None, :]

        # Spatial 3D conditioning
        c_cond = nn.Conv(2 * self.out_ch, kernel_size=(3, 3, 3), padding="SAME")(cond_xyz)
        c_scale, c_shift = jnp.split(c_cond, 2, axis=-1)

        h = h * (1.0 + t_scale + c_scale) + (t_shift + c_shift)
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
        x = jax.image.resize(
            x,
            shape=(B, Z * 2, Y * 2, X * 2, C),
            method="nearest",
        )
        x = nn.Conv(self.out_ch, kernel_size=(3, 3, 3), padding="SAME")(x)
        return x


class Conditioning2DEncoder(nn.Module):
    channels: Sequence[int] = (32, 64, 128)
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, images):
        """
        images: (B, H, W, C)
        returns multiscale 2D features
        """
        feats = []

        x = nn.Conv(self.channels[0], kernel_size=(3, 3), padding="SAME")(images)
        x = self.activation(x)
        feats.append(x)

        for ch in self.channels[1:]:
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
            x = nn.Conv(ch, kernel_size=(3, 3), padding="SAME")(x)
            x = self.activation(x)
            feats.append(x)

        return feats


def make_3d_coord_grid(zdim: int, ydim: int, xdim: int) -> jnp.ndarray:
    """
    Returns (Z, Y, X, 3) with coordinates in [-1, 1].
    """
    z = jnp.linspace(-1.0, 1.0, zdim, dtype=jnp.float32)
    y = jnp.linspace(-1.0, 1.0, ydim, dtype=jnp.float32)
    x = jnp.linspace(-1.0, 1.0, xdim, dtype=jnp.float32)
    zz, yy, xx = jnp.meshgrid(z, y, x, indexing="ij")
    return jnp.stack([xx, yy, zz], axis=-1)


def make_1d_z_coords(zdim: int) -> jnp.ndarray:
    return jnp.linspace(-1.0, 1.0, zdim, dtype=jnp.float32)


class Lift2DTo3D(nn.Module):
    out_ch: int
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, feat2d: jnp.ndarray, zdim: int) -> jnp.ndarray:
        """
        feat2d:  (B, Y, X, C)
        returns: (B, Z, Y, X, out_ch)
        """
        B, Y, X, C = feat2d.shape

        tiled = jnp.repeat(feat2d[:, None, :, :, :], repeats=zdim, axis=1)  # (B,Z,Y,X,C)

        zcoords = make_1d_z_coords(zdim)[:, None]  # (Z,1)
        zemb = MLP([self.out_ch, self.out_ch], activate_last=True)(zcoords)  # (Z,out_ch)
        zemb = zemb[None, :, None, None, :]  # (1,Z,1,1,out_ch)
        zemb = jnp.broadcast_to(zemb, (B, zdim, Y, X, self.out_ch))

        h = jnp.concatenate([tiled, zemb], axis=-1)
        h = nn.Conv(self.out_ch, kernel_size=(3, 3, 3), padding="SAME")(h)
        h = self.activation(h)
        h = nn.Conv(self.out_ch, kernel_size=(3, 3, 3), padding="SAME")(h)
        h = self.activation(h)
        return h


def sample_2d_features_at_pixels(
    feat2d: jnp.ndarray,
    pixel_coords: jnp.ndarray,
    coord_image_size: int,
) -> jnp.ndarray:
    """
    Nearest-neighbor sampling.

    feat2d:          (B, H, W, C)
    pixel_coords:    (B, N, 2) in coordinates defined on the original image grid
    coord_image_size: size of that original image grid, e.g. 16 or 64

    returns:         (B, N, C)
    """
    B, H, W, C = feat2d.shape
    _, N, _ = pixel_coords.shape

    x = pixel_coords[..., 0]
    y = pixel_coords[..., 1]

    denom = float(max(coord_image_size - 1, 1))
    x = x * ((W - 1) / denom)
    y = y * ((H - 1) / denom)

    x_idx = jnp.clip(jnp.rint(x).astype(jnp.int32), 0, W - 1)
    y_idx = jnp.clip(jnp.rint(y).astype(jnp.int32), 0, H - 1)

    b_idx = jnp.arange(B, dtype=jnp.int32)[:, None]
    sampled = feat2d[b_idx, y_idx, x_idx]  # (B,N,C)
    return sampled


class GalaxyTokenEncoder(nn.Module):
    token_dim: int
    coord_image_size: int
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(
        self,
        gal_features: jnp.ndarray,
        gal_pixel_coords: jnp.ndarray,
        image_feats_2d: Sequence[jnp.ndarray],
        gal_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        gal_features:    (B, N, Fg)   e.g. [x, y, vz, Ngal]
        gal_pixel_coords:(B, N, 2)
        gal_mask:        (B, N)

        returns:
            galaxy tokens: (B, N, token_dim)
        """
        sampled_feats = [
            sample_2d_features_at_pixels(feat, gal_pixel_coords, self.coord_image_size)
            for feat in image_feats_2d
        ]
        img_local = jnp.concatenate(sampled_feats, axis=-1)  # (B,N,sumC)

        denom = float(max(self.coord_image_size - 1, 1))
        pix_norm = gal_pixel_coords / denom
        pix_norm = 2.0 * pix_norm - 1.0

        x = jnp.concatenate([gal_features, pix_norm, img_local], axis=-1)
        x = MLP([self.token_dim, self.token_dim, self.token_dim], activate_last=True)(x)

        x = x * gal_mask[..., None]
        return x


class QueryCrossAttention3D(nn.Module):
    query_dim: int
    num_heads: int
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(
        self,
        galaxy_tokens: jnp.ndarray,
        gal_mask: jnp.ndarray,
        time_emb: jnp.ndarray,
        zdim: int,
        ydim: int,
        xdim: int,
    ) -> jnp.ndarray:
        """
        galaxy_tokens: (B, N, Dtok)
        gal_mask:      (B, N)
        time_emb:      (B, T)

        returns:
            cond_latent: (B, Z, Y, X, query_dim)
        """
        B, N, _ = galaxy_tokens.shape

        coords = make_3d_coord_grid(zdim, ydim, xdim)       # (Z,Y,X,3)
        q_pos = coords.reshape(-1, 3)                       # (Q,3)
        Q = q_pos.shape[0]

        base_query = self.param(
            "base_query",
            nn.initializers.normal(stddev=0.02),
            (Q, self.query_dim),
        )

        q_pos_emb = MLP([self.query_dim, self.query_dim], activate_last=True)(q_pos)  # (Q,D)

        t_proj = nn.Dense(self.query_dim)(time_emb)         # (B,D)
        t_proj = t_proj[:, None, :]                         # (B,1,D)

        queries = base_query[None, :, :] + q_pos_emb[None, :, :] + t_proj  # (B,Q,D)

        attn_mask = nn.make_attention_mask(
            jnp.ones((B, Q), dtype=jnp.float32),
            gal_mask.astype(jnp.float32),
        )  # (B,1,Q,N)

        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.query_dim,
            out_features=self.query_dim,
            dropout_rate=0.0,
            deterministic=True,
        )(queries, galaxy_tokens, mask=attn_mask)

        h = nn.LayerNorm()(h + queries)
        mlp_out = MLP([2 * self.query_dim, self.query_dim], activate_last=False)(h)
        h = nn.LayerNorm()(h + mlp_out)

        h = h.reshape(B, zdim, ydim, xdim, self.query_dim)
        return h


class FusionBlock3D(nn.Module):
    out_ch: int
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        a, b: (B, Z, Y, X, C)
        """
        h = jnp.concatenate([a, b], axis=-1)
        h = nn.Conv(self.out_ch, kernel_size=(3, 3, 3), padding="SAME")(h)
        h = self.activation(h)
        h = nn.Conv(self.out_ch, kernel_size=(3, 3, 3), padding="SAME")(h)
        h = self.activation(h)
        return h


class ConditioningPyramid3D(nn.Module):
    channels: Sequence[int]
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, cond0: jnp.ndarray):
        """
        Build multiscale conditioning volumes from a full-resolution fused volume.

        cond0: (B, Z, Y, X, channels[0])

        returns:
            [cond0, cond1, cond2, ...]
        """
        feats = [cond0]
        x = cond0
        for ch in self.channels[1:]:
            x = Downsample3D(ch)(x)
            x = self.activation(x)
            x = nn.Conv(ch, kernel_size=(3, 3, 3), padding="SAME")(x)
            x = self.activation(x)
            feats.append(x)
        return feats


@dataclass
class DiffusionModelConfig:
    base_channels: int = 32
    channel_mults: Sequence[int] = (1, 2, 4)
    time_emb_dim: int = 128
    out_channels: int = 1

    galaxy_token_dim: int = 128
    num_attention_heads: int = 4

    # Pixel coordinate system size for gal_pixel_coords
    # Use 16 for your new low-resolution dataset, 64 for the old one.
    coord_image_size: int = 16


class ConditionalUNet3D(nn.Module):
    cfg: DiffusionModelConfig

    @nn.compact
    def __call__(
        self,
        noisy_cube: jnp.ndarray,
        timesteps: jnp.ndarray,
        cond_images: jnp.ndarray,
        gal_features: jnp.ndarray,
        gal_pixel_coords: jnp.ndarray,
        gal_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        noisy_cube:       (B, Z, Y, X, 1)      e.g. (B,16,16,16,1)
        timesteps:        (B,)
        cond_images:      (B, H, W, 4)         e.g. (B,16,16,4)
        gal_features:     (B, N, Fg)           e.g. [x, y, vz, Ngal]
        gal_pixel_coords: (B, N, 2)
        gal_mask:         (B, N)
        """
        time_emb = sinusoidal_time_embedding(timesteps, self.cfg.time_emb_dim)
        time_emb = MLP(
            [self.cfg.time_emb_dim, self.cfg.time_emb_dim],
            activate_last=True,
        )(time_emb)

        chs = [self.cfg.base_channels * m for m in self.cfg.channel_mults]

        # ------------------------------------------------------------
        # 2D image branch
        # ------------------------------------------------------------
        cond_feats_2d = Conditioning2DEncoder(channels=chs)(cond_images)
        # For 16x16 input and chs=[32,64,128]:
        # cond_feats_2d = [16x16x32, 8x8x64, 4x4x128]

        # ------------------------------------------------------------
        # Galaxy token branch
        # ------------------------------------------------------------
        galaxy_tokens = GalaxyTokenEncoder(
            token_dim=self.cfg.galaxy_token_dim,
            coord_image_size=self.cfg.coord_image_size,
        )(
            gal_features=gal_features,
            gal_pixel_coords=gal_pixel_coords,
            image_feats_2d=cond_feats_2d,
            gal_mask=gal_mask,
        )  # (B,N,Dtok)

        # ------------------------------------------------------------
        # Full-resolution conditioning from image branch
        # ------------------------------------------------------------
        z0, y0, x0 = noisy_cube.shape[1:4]
        img_cond0 = Lift2DTo3D(out_ch=chs[0])(cond_feats_2d[0], zdim=z0)  # (B,Z,Y,X,chs[0])

        # ------------------------------------------------------------
        # Full-resolution conditioning from galaxy branch
        # ------------------------------------------------------------
        gal_cond0 = QueryCrossAttention3D(
            query_dim=chs[0],
            num_heads=self.cfg.num_attention_heads,
        )(
            galaxy_tokens=galaxy_tokens,
            gal_mask=gal_mask,
            time_emb=time_emb,
            zdim=z0,
            ydim=y0,
            xdim=x0,
        )  # (B,Z,Y,X,chs[0])

        # ------------------------------------------------------------
        # Fuse the two conditioning volumes BEFORE the U-Net downsampling
        # ------------------------------------------------------------
        cond0 = FusionBlock3D(out_ch=chs[0])(gal_cond0, img_cond0)  # full-res fused conditioning

        # Build conditioning pyramid by downsampling the fused full-res volume
        cond_volumes = ConditioningPyramid3D(channels=chs)(cond0)
        # Example for 16^3:
        # cond_volumes = [16^3x32, 8^3x64, 4^3x128]

        # ------------------------------------------------------------
        # 3D U-Net
        # ------------------------------------------------------------
        x = nn.Conv(chs[0], kernel_size=(3, 3, 3), padding="SAME")(noisy_cube)

        skips = []
        for level, ch in enumerate(chs[:-1]):
            cond_xyz = cond_volumes[level]

            x = ResBlock3D(ch, time_dim=self.cfg.time_emb_dim)(x, time_emb, cond_xyz)
            x = ResBlock3D(ch, time_dim=self.cfg.time_emb_dim)(x, time_emb, cond_xyz)
            skips.append(x)

            x = Downsample3D(chs[level + 1])(x)

        bottleneck_cond = cond_volumes[-1]
        x = ResBlock3D(chs[-1], time_dim=self.cfg.time_emb_dim)(x, time_emb, bottleneck_cond)
        x = ResBlock3D(chs[-1], time_dim=self.cfg.time_emb_dim)(x, time_emb, bottleneck_cond)

        for level, (ch, skip) in enumerate(zip(reversed(chs[:-1]), reversed(skips))):
            x = Upsample3D(ch)(x)

            min_z = min(x.shape[1], skip.shape[1])
            min_y = min(x.shape[2], skip.shape[2])
            min_x = min(x.shape[3], skip.shape[3])

            x = x[:, :min_z, :min_y, :min_x, :]
            skip = skip[:, :min_z, :min_y, :min_x, :]

            x = jnp.concatenate([x, skip], axis=-1)

            cond_xyz = cond_volumes[len(chs[:-1]) - 1 - level]
            cond_xyz = cond_xyz[:, :min_z, :min_y, :min_x, :]

            x = ResBlock3D(ch, time_dim=self.cfg.time_emb_dim)(x, time_emb, cond_xyz)
            x = ResBlock3D(ch, time_dim=self.cfg.time_emb_dim)(x, time_emb, cond_xyz)

        x = nn.GroupNorm(num_groups=min(8, x.shape[-1]))(x)
        x = nn.gelu(x)
        x = nn.Conv(self.cfg.out_channels, kernel_size=(3, 3, 3), padding="SAME")(x)
        return x