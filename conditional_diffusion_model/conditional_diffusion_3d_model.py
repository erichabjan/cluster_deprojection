from dataclasses import dataclass
from typing import Sequence, Callable, Optional

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


class SourceShapeEncoder(nn.Module):
    """
    Weak-lensing source-galaxy branch (replaces the projected mass image).

    Per-galaxy MLP embedding of the 7 source features -> mean-pooled per cell
    of the (grid_size x grid_size) shape grid via segment ops (rows arrive
    pre-sorted by cell id from the dataset) -> concat a log1p(count) channel
    -> stride-2 conv stages down to (target_size x target_size, out_ch).

    Padded galaxies must carry cell id = grid_size**2; they land in a dummy
    segment that is dropped, so no separate validity mask is needed.
    """
    embed_dim: int          # per-galaxy embedding length (cfg.src_embed_dim)
    out_ch: int             # channels of the reduced feature map
    grid_size: int          # S: shape grid resolution (cfg.shape_grid_size)
    target_size: int        # output resolution, matches the conditioning images
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(
        self,
        src_features: jnp.ndarray,
        src_cell_id: jnp.ndarray,
        src_log_count: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        src_features:  (B, N, 7)  e1, e2, ix, iy, z_s, z_lens, cell_frac (raw)
        src_cell_id:   (B, N)     int32, iy*S + ix; padded rows = S*S
        src_log_count: (B, S, S, 1)  log1p(sources per cell)

        returns: (B, target_size, target_size, out_ch)
        """
        B, N, _ = src_features.shape
        S = self.grid_size
        n_seg = S * S + 1  # + dummy segment for padded rows

        # Fixed rescaling so all inputs are O(1): cell coords -> [-1, 1],
        # cell_frac -> ~1 for a uniform field (raw values are ~1/S^2).
        f = src_features
        x_in = jnp.stack([
            f[..., 0],                                     # e1
            f[..., 1],                                     # e2
            (f[..., 2] + 0.5) / S * 2.0 - 1.0,             # ix -> [-1, 1]
            (f[..., 3] + 0.5) / S * 2.0 - 1.0,             # iy -> [-1, 1]
            f[..., 4],                                     # z_s
            f[..., 5],                                     # z_lens
            f[..., 6] * float(S * S),                      # cell_frac
        ], axis=-1)

        emb = MLP([2 * self.embed_dim, self.embed_dim], activate_last=True)(x_in)  # (B,N,E)

        def cell_mean(e, cid):
            sums = jax.ops.segment_sum(
                e, cid, num_segments=n_seg, indices_are_sorted=True)
            cnts = jax.ops.segment_sum(
                jnp.ones((e.shape[0],), e.dtype), cid,
                num_segments=n_seg, indices_are_sorted=True)
            return sums[:-1] / jnp.maximum(cnts[:-1], 1.0)[:, None]  # drop dummy

        h = jax.vmap(cell_mean)(emb, src_cell_id)          # (B, S*S, E)
        h = h.reshape(B, S, S, self.embed_dim)             # [iy, ix] = (H, W)
        h = jnp.concatenate([h, src_log_count], axis=-1)   # (B, S, S, E+1)

        size = S
        assert size % self.target_size == 0, \
            f"shape grid {S} not reducible to {self.target_size} by stride-2 convs"
        while size > self.target_size:
            h = nn.Conv(self.out_ch, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(h)
            h = nn.GroupNorm(num_groups=min(8, self.out_ch))(h)
            h = self.activation(h)
            size //= 2

        h = nn.Conv(self.out_ch, kernel_size=(3, 3), padding="SAME")(h)
        h = self.activation(h)
        return h


def _pixel_overlap_matrix(src_pix: jnp.ndarray, n_src: int, n_dst: int,
                          fov: float) -> jnp.ndarray:
    """
    1D overlap lengths between a centred source grid and the physical grid.

    src_pix: (B,) source pixel size (Mpc). The source grid has n_src pixels
             centred on 0: pixel i spans [(i - n_src/2) s, (i + 1 - n_src/2) s].
    The destination grid has n_dst pixels over [-fov, fov].

    returns: (B, n_dst, n_src); entry [b, j, i] is the length (Mpc) shared by
             source pixel i and destination pixel j.
    """
    e = jnp.arange(n_src + 1, dtype=jnp.float32) - 0.5 * n_src
    src_lo = e[None, :-1] * src_pix[:, None]                      # (B,n_src)
    src_hi = e[None, 1:] * src_pix[:, None]
    d = jnp.linspace(-fov, fov, n_dst + 1, dtype=jnp.float32)
    lo = jnp.maximum(src_lo[:, None, :], d[None, :-1, None])      # (B,n_dst,n_src)
    hi = jnp.minimum(src_hi[:, None, :], d[None, 1:, None])
    return jnp.maximum(hi - lo, 0.0)


def reproject_xray_to_physical(
    values: jnp.ndarray,
    observed: jnp.ndarray,
    mpc_per_pix: jnp.ndarray,
    out_size: int,
    fov: float,
):
    """
    Exact, area-weighted reprojection of angular X-ray maps onto the physical
    (out_size x out_size) grid over [-fov, fov] Mpc. Deterministic and pure
    jnp, so it jits, vmaps and differentiates.

    Both grids are centred on the BCG and share the viewing frame's axes (the
    dataset's xray_utils.to_frame_grid), so the only transformation is a
    per-sample scale, and the 2D pixel-overlap areas factor into the 1D
    overlaps of the two axes: area[j, k, y, x] = W[j, y] W[k, x]. The values
    are averaged over the *observed* part of each output pixel only, so an
    edge pixel half on the detector is not diluted by the half that is off it.

    values:      (B, G, G, C)  rows are y, columns x, as in the dataset
    observed:    (B, G, G)     1 where the detector looked (expmap > 0)
    mpc_per_pix: (B,)          physical size of a source pixel

    returns:
      mean  (B, T, T, C)  area-weighted mean over observed area; 0 where none
      f_obs (B, T, T)     observed fraction of each output pixel's area
    """
    G = values.shape[1]
    W = _pixel_overlap_matrix(mpc_per_pix, G, out_size, fov)       # (B,T,G)
    obs = observed.astype(values.dtype)
    num = jnp.einsum("bjy,byxc,bkx->bjkc", W, values * obs[..., None], W)
    den = jnp.einsum("bjy,byx,bkx->bjk", W, obs, W)
    mean = num / jnp.maximum(den, 1e-30)[..., None]
    f_obs = den / (2.0 * fov / out_size) ** 2
    return mean, f_obs


class XrayEncoder(nn.Module):
    """
    Mock-Chandra X-ray branch.

    The maps live on a fixed *angular* grid (a 20' ACIS-I field, 128^2), while
    every other input is on the physical +/-fov_mpc grid; the X-ray field is
    0.7-2.6 Mpc across depending on z_lens. So the maps are first reprojected,
    exactly and deterministically, onto a physical (reproj_size)^2 grid over
    +/-fov_mpc (reproject_xray_to_physical), then strided down to target_size
    by convolutions, as in SourceShapeEncoder.

    Every output pixel is in one of three states, each with its own value so
    the convolutions can tell them apart:

      detected                xray_mask = 1          counts_z as stored (>= -1:
                                                     one count is -1.0)
      observed, no detection  expmap > 0, mask = 0   counts_z = nodet_counts_z
      unobserved              off the detector       counts_z = unobs_counts_z,
                              footprint (expmap = 0)  exposure = unobs_exposure

    The dataset stores counts_z = 0 on every pixel without a detection -- a
    filler meant to be ignored via the mask, but 0 is a real detected value
    (~3 counts) -- so non-detections are rewritten to nodet_counts_z here,
    *before* reprojection, from xray_mask.

    'Unobserved' is the detector footprint, not the 20' square: ~24% of that
    square is off ACIS-I (expmap = 0) in every sample, and on the physical grid
    everything outside the field is unobserved too. An output pixel is
    unobserved only if none of its area was observed; partly observed pixels
    carry their observed fraction as a channel.
    """
    out_ch: int             # channels of the reduced feature map
    grid_size: int          # G: Chandra grid resolution (cfg.xray_grid_size)
    reproj_size: int        # physical grid the maps are reprojected onto
    target_size: int        # output resolution, matches the conditioning images
    fov_mpc: float          # half-extent of the physical grids (cfg.map_fov_mpc)
    nodet_counts_z: float = -2.0
    unobs_counts_z: float = -5.0
    unobs_exposure: float = -1.0
    exposure_ref_cm2: float = 104.0   # max expmap in the datasets, so exposure is (0, 1]
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(
        self,
        xray_maps: jnp.ndarray,
        xray_scalars: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        xray_maps:    (B, G, G, 3)  counts_z, xray_mask, expmap -- as stored:
                                    counts log10 z-scored, mask 0/1, expmap cm^2
        xray_scalars: (B, 3)        z_lens, log10 t_exp_ks, log10 mpc_per_pix (raw)

        returns: (B, target_size, target_size, out_ch)
        """
        B, G = xray_maps.shape[0], xray_maps.shape[1]
        assert G == self.grid_size, f"X-ray maps are {G}^2, config says {self.grid_size}^2"
        R, T = self.reproj_size, self.target_size
        assert R % T == 0 and ((R // T) & (R // T - 1)) == 0, \
            f"reprojection grid {R} not reducible to {T} by stride-2 convs"

        counts_z = xray_maps[..., 0]
        detected = xray_maps[..., 1] > 0.5
        exposure = xray_maps[..., 2]
        observed = exposure > 0.0

        counts_z = jnp.where(detected, counts_z, self.nodet_counts_z)
        exposure = exposure / self.exposure_ref_cm2
        values = jnp.stack([counts_z, exposure], axis=-1)            # (B,G,G,2)

        mpc_per_pix = 10.0 ** xray_scalars[:, 2]
        mean, f_obs = reproject_xray_to_physical(
            values, observed, mpc_per_pix, R, self.fov_mpc)          # (B,R,R,2), (B,R,R)

        obs_out = f_obs > 1e-6
        counts_r = jnp.where(obs_out, mean[..., 0], self.unobs_counts_z)
        exposure_r = jnp.where(obs_out, mean[..., 1], self.unobs_exposure)

        scal = jnp.stack([
            (xray_scalars[:, 0] - 0.075) / 0.026,     # z_lens
            (xray_scalars[:, 1] - 1.5) / 0.25,        # log10 t_exp_ks
            (xray_scalars[:, 2] + 2.0) / 0.3,         # log10 mpc_per_pix
        ], axis=-1)                                                   # (B,3)

        h = jnp.concatenate([
            counts_r[..., None],
            exposure_r[..., None],
            obs_out.astype(counts_r.dtype)[..., None],
            jnp.clip(f_obs, 0.0, 1.0)[..., None],
            jnp.broadcast_to(scal[:, None, None, :], (B, R, R, 3)),
        ], axis=-1)                                                   # (B,R,R,7)

        h = nn.Conv(self.out_ch, kernel_size=(3, 3), padding="SAME")(h)
        h = self.activation(h)
        size = R
        while size > T:
            h = nn.Conv(self.out_ch, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(h)
            h = nn.GroupNorm(num_groups=min(8, self.out_ch))(h)
            h = self.activation(h)
            size //= 2

        h = nn.Conv(self.out_ch, kernel_size=(3, 3), padding="SAME")(h)
        h = self.activation(h)
        return h


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

    # Weak-lensing source-galaxy branch (shape_dynamics dataset)
    src_embed_dim: int = 32       # per-source-galaxy embedding length
    src_out_ch: int = 32          # channels of the reduced shear feature map
    shape_grid_size: int = 128    # S: source cell grid (h5 attr 'shape_grid_size')

    # Mock-Chandra X-ray branch (shape_dynamics dataset)
    xray_out_ch: int = 32         # channels of the reduced xray feature map
    xray_grid_size: int = 128     # G: Chandra grid (h5 attr 'xray_grid_size')
    xray_reproj_size: int = 128   # physical grid the maps are reprojected onto
    # Values of the three pixel states (see XrayEncoder). Detected counts_z are
    # >= -1.0 (one count) in every dataset, exposure / ref is in (0, 1].
    xray_nodet_counts_z: float = -2.0
    xray_unobs_counts_z: float = -5.0
    xray_unobs_exposure: float = -1.0
    xray_exposure_ref_cm2: float = 104.0
    # Half-extent of the conditioning/shape grids, in Mpc (h5 attr
    # 'map_fov_mpc'). The X-ray branch needs it because its own grid is
    # angular: it sets the physical grid the maps are reprojected onto.
    map_fov_mpc: float = 5.0


class ConditionalUNet3D(nn.Module):
    cfg: DiffusionModelConfig

    @nn.compact
    def __call__(
        self,
        noisy_cube: jnp.ndarray,
        timesteps: jnp.ndarray,
        cond_images: Optional[jnp.ndarray] = None,
        gal_features: Optional[jnp.ndarray] = None,
        gal_pixel_coords: Optional[jnp.ndarray] = None,
        gal_mask: Optional[jnp.ndarray] = None,
        src_features: Optional[jnp.ndarray] = None,
        src_cell_id: Optional[jnp.ndarray] = None,
        src_log_count: Optional[jnp.ndarray] = None,
        xray_maps: Optional[jnp.ndarray] = None,
        xray_scalars: Optional[jnp.ndarray] = None,
        cond_cache: Optional[dict] = None,
        encode_only: bool = False,
    ):
        """
        noisy_cube:       (B, Z, Y, X, 1)      e.g. (B,16,16,16,1)
        timesteps:        (B,)
        cond_images:      (B, H, W, C)         e.g. (B,16,16,3)
        gal_features:     (B, N, Fg)           e.g. [x, y, vz, Ngal]
        gal_pixel_coords: (B, N, 2)
        gal_mask:         (B, N)
        src_features:     (B, Ns, 7)           e1, e2, ix, iy, z_s, z_lens, cell_frac
        src_cell_id:      (B, Ns)              int32; padded rows = shape_grid_size**2
        src_log_count:    (B, S, S, 1)         log1p(sources per shape-grid cell)
        xray_maps:        (B, G, G, 3)         counts_z, xray_mask, expmap (as
                                               stored); None disables the branch
        xray_scalars:     (B, 3)               z_lens, log10 t_exp_ks, log10 mpc_per_pix

        Sampling shortcut. Everything up to the galaxy tokens and the lifted
        image volume depends only on the conditioning, not on noisy_cube's
        values or timesteps, so a sampler can compute it once per batch:
          encode_only=True  returns that dict (cond_cache) instead of the
                            noise prediction; noisy_cube supplies only the
                            cube shape and timesteps are unused
          cond_cache=dict   skips those branches and reuses the dict; the
                            conditioning arguments are then not needed
        Training and init pass neither, which runs the full forward pass. The
        parameter tree is identical either way (submodule names do not
        depend on which branches run), so existing checkpoints load as is.
        """
        chs = [self.cfg.base_channels * m for m in self.cfg.channel_mults]
        z0, y0, x0 = noisy_cube.shape[1:4]

        if cond_cache is None:
            # ------------------------------------------------------------
            # Weak-lensing source-galaxy branch -> 2D feature map fused
            # channel-wise into the conditioning images (the slot the
            # projected mass map used to occupy)
            # ------------------------------------------------------------
            shear2d = SourceShapeEncoder(
                embed_dim=self.cfg.src_embed_dim,
                out_ch=self.cfg.src_out_ch,
                grid_size=self.cfg.shape_grid_size,
                target_size=cond_images.shape[1],
            )(src_features, src_cell_id, src_log_count)     # (B,H,W,src_out_ch)

            # ------------------------------------------------------------
            # Mock-Chandra X-ray branch -> the same 2D feature map. Optional,
            # so a dynamics/lensing-only dataset still runs. The angular maps
            # are reprojected onto the physical grid first, then strided down
            # like the branch above; see XrayEncoder.
            # ------------------------------------------------------------
            extra_2d = [shear2d]
            if xray_maps is not None:
                xray2d = XrayEncoder(
                    out_ch=self.cfg.xray_out_ch,
                    grid_size=self.cfg.xray_grid_size,
                    reproj_size=self.cfg.xray_reproj_size,
                    target_size=cond_images.shape[1],
                    fov_mpc=self.cfg.map_fov_mpc,
                    nodet_counts_z=self.cfg.xray_nodet_counts_z,
                    unobs_counts_z=self.cfg.xray_unobs_counts_z,
                    unobs_exposure=self.cfg.xray_unobs_exposure,
                    exposure_ref_cm2=self.cfg.xray_exposure_ref_cm2,
                )(xray_maps, xray_scalars)                  # (B,H,W,xray_out_ch)
                extra_2d.append(xray2d)

            cond_images = jnp.concatenate([cond_images, *extra_2d], axis=-1)

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
            img_cond0 = Lift2DTo3D(out_ch=chs[0])(cond_feats_2d[0], zdim=z0)  # (B,Z,Y,X,chs[0])

            cond_cache = dict(galaxy_tokens=galaxy_tokens, gal_mask=gal_mask,
                              img_cond0=img_cond0)
            if encode_only:
                return cond_cache

        galaxy_tokens = cond_cache["galaxy_tokens"]
        gal_mask = cond_cache["gal_mask"]
        img_cond0 = cond_cache["img_cond0"]

        time_emb = sinusoidal_time_embedding(timesteps, self.cfg.time_emb_dim)
        time_emb = MLP(
            [self.cfg.time_emb_dim, self.cfg.time_emb_dim],
            activate_last=True,
        )(time_emb)

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