from dataclasses import dataclass
from typing import Sequence, Callable

import jax.numpy as jnp
import flax.linen as nn


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


class Conditioning2DEncoder(nn.Module):
    channels: Sequence[int] = (32, 64, 128)
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, images):
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


def sample_2d_features_at_pixels(
    feat2d: jnp.ndarray,
    pixel_coords: jnp.ndarray,
    coord_image_size: int,
) -> jnp.ndarray:
    """
    Nearest-neighbor sampling of a 2D feature map at galaxy pixel locations.

    feat2d:           (B, H, W, C)
    pixel_coords:     (B, N, 2) in coordinates defined on the original image grid
    coord_image_size: size of that original image grid

    returns:          (B, N, C)
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
    sampled = feat2d[b_idx, y_idx, x_idx]
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
        gal_features:     (B, N, Fg)   e.g. [x, y, vz, Ngal]
        gal_pixel_coords: (B, N, 2)
        gal_mask:         (B, N)

        returns:
            galaxy tokens: (B, N, token_dim)
        """
        sampled_feats = [
            sample_2d_features_at_pixels(feat, gal_pixel_coords, self.coord_image_size)
            for feat in image_feats_2d
        ]
        img_local = jnp.concatenate(sampled_feats, axis=-1)

        denom = float(max(self.coord_image_size - 1, 1))
        pix_norm = gal_pixel_coords / denom
        pix_norm = 2.0 * pix_norm - 1.0

        x = jnp.concatenate([gal_features, pix_norm, img_local], axis=-1)
        x = MLP([self.token_dim, self.token_dim, self.token_dim], activate_last=True)(x)

        x = x * gal_mask[..., None]
        return x


class GlobalQueryAttentionPool(nn.Module):
    """
    K learned global queries cross-attend over the galaxy tokens to produce
    a fixed-size pooled representation.
    """
    num_queries: int
    query_dim: int
    num_heads: int
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(
        self,
        galaxy_tokens: jnp.ndarray,
        gal_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        galaxy_tokens: (B, N, Dtok)
        gal_mask:      (B, N)

        returns:       (B, K, query_dim)
        """
        B, N, _ = galaxy_tokens.shape
        K = self.num_queries

        base_query = self.param(
            "global_query",
            nn.initializers.normal(stddev=0.02),
            (K, self.query_dim),
        )
        queries = jnp.broadcast_to(base_query[None, :, :], (B, K, self.query_dim))

        attn_mask = nn.make_attention_mask(
            jnp.ones((B, K), dtype=jnp.float32),
            gal_mask.astype(jnp.float32),
        )

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
        return h


@dataclass
class GlobalsModelConfig:
    base_channels: int = 32
    channel_mults: Sequence[int] = (1, 2, 4)

    galaxy_token_dim: int = 128
    num_attention_heads: int = 4
    num_global_queries: int = 8

    coord_image_size: int = 16

    head_hidden: Sequence[int] = (256, 256)
    out_dim: int = 4  # [mass_log10, axis_a, axis_b, axis_c] (standardized)


class CNNSetGlobalsModel(nn.Module):
    """
    Point estimator for global cluster quantities (cube enclosed mass and
    three shape-tensor axis lengths). Reuses the 2D image encoder and galaxy
    token encoder from the diffusion model, then pools galaxy tokens with a
    small set of learned global queries via cross-attention.
    """
    cfg: GlobalsModelConfig

    @nn.compact
    def __call__(
        self,
        cond_images: jnp.ndarray,
        gal_features: jnp.ndarray,
        gal_pixel_coords: jnp.ndarray,
        gal_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        cond_images:      (B, H, W, C)        e.g. (B, 16, 16, 4)
        gal_features:     (B, N, Fg)          e.g. [x, y, vz, Ngal]
        gal_pixel_coords: (B, N, 2)
        gal_mask:         (B, N)

        returns:          (B, out_dim)
        """
        chs = [self.cfg.base_channels * m for m in self.cfg.channel_mults]

        cond_feats_2d = Conditioning2DEncoder(channels=chs)(cond_images)

        galaxy_tokens = GalaxyTokenEncoder(
            token_dim=self.cfg.galaxy_token_dim,
            coord_image_size=self.cfg.coord_image_size,
        )(
            gal_features=gal_features,
            gal_pixel_coords=gal_pixel_coords,
            image_feats_2d=cond_feats_2d,
            gal_mask=gal_mask,
        )

        pooled_tokens = GlobalQueryAttentionPool(
            num_queries=self.cfg.num_global_queries,
            query_dim=self.cfg.galaxy_token_dim,
            num_heads=self.cfg.num_attention_heads,
        )(galaxy_tokens, gal_mask)
        # (B, K, Dtok) -> (B, K * Dtok)
        B = pooled_tokens.shape[0]
        pooled_flat = pooled_tokens.reshape(B, -1)

        # Global image context from the deepest feature map.
        deep_feat = cond_feats_2d[-1]
        img_mean = jnp.mean(deep_feat, axis=(1, 2))
        img_max = jnp.max(deep_feat, axis=(1, 2))
        img_pool = jnp.concatenate([img_mean, img_max], axis=-1)

        h = jnp.concatenate([pooled_flat, img_pool], axis=-1)
        h = MLP(
            [*self.cfg.head_hidden, self.cfg.out_dim],
            activate_last=False,
        )(h)
        return h
