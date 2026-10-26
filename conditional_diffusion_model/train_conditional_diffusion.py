from typing import Dict, Iterator, Tuple, Optional, Sequence
import time
import h5py
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

import os
import pickle
from flax import serialization

import wandb


# Nominal standardization of the density cubes, shared by every suite and fixed
# rather than measured, so a model's output can always be turned back into a
# density: log10(rho / Msun Mpc^-3) = z * CUBE_LOG10_STD + CUBE_LOG10_MEAN.
# Each dataset file stores its cubes z-scored with its own stats
# (cube_log10_{mean,std}_N); load_datasets converts them to these.
CUBE_LOG10_MEAN = 11.0
CUBE_LOG10_STD = 1.0
CUBE_FLOOR_VALUE = -5.0     # empty voxels (rho = 0), as in the dataset files

# Source-catalog columns the model's SourceShapeEncoder consumes: e1, e2, ix,
# iy, z_s, z_lens, cell_frac. The files also carry an 8th, coverage, which the
# model does not use yet.
N_SRC_FEATURES = 7


def _suite_name(f: h5py.File, path: str) -> str:
    """The file's simulation suite: its attr when written, else the file name."""
    if "simulation_suite" in f.attrs:
        return str(f.attrs["simulation_suite"])
    return os.path.basename(path).split("_")[0].upper()


def load_datasets(file_paths: Sequence[str], resolution: int,
                  keys: Optional[Sequence[Optional[Sequence[str]]]] = None) -> Dict[str, np.ndarray]:
    """
    Load and pool several dataset files (e.g. BAHAMAS + FLAMINGO) at one
    resolution N into memory. Only the _N datasets are read.

    keys, if given, holds one entry per file: the sample group names to read
    from it (e.g. ["000000", "000008"]), or None for all of them. The default
    reads every sample of every file.

    Dense fields are pre-allocated over all files and filled in place, so the
    pooled set costs one copy (~55 GB at N = 128 for 5000 samples). Train/val
    splits are index arrays into it (split_by_cluster), never copies.

      images           (n, 3, N, N)    images_N
      cubes            (n, N, N, N)    density_cube_N, re-standardized from each
                                       file's stats to CUBE_LOG10_{MEAN,STD};
                                       empty voxels stay at CUBE_FLOOR_VALUE
      gal_features     (n, max_nodes, 4)
      gal_pixel_coords (n, max_nodes, 2) gal_pixel_coords_N, on the N x N grid
      mask             (n, max_nodes)
      globals_phys     (n, 4)          log10 cube mass [Msun], axis a, b, c [Mpc]:
                                       physical, so comparable across suites
      suite, simulation, cluster_index, projection_index   per-sample labels

    The weak-lensing source point clouds are RAGGED (variable count per
    sample) and are lists of arrays:
      src_features [i]: (N_i, 7) float32  e1, e2, ix, iy, z_s, z_lens, cell_frac
      src_cell_id  [i]: (N_i,)   int32    iy*S + ix, rows pre-sorted by this
      src_cell_ptr     : (n, S*S+1) int64 CSR pointers; sample i's cell c
                         holds src_features[i][ptr[i,c]:ptr[i,c+1]]
    with S = metadata["shape_grid_size"].

    The mock-Chandra maps are fixed-shape, so they stack densely:
      xray_maps    : (n, G, G, 3) float32, channels
                     [counts_z, xray_mask, expmap], exactly as stored
      xray_scalars : (n, 3) float32, [z_lens, log10 t_exp_ks,
                     log10 xray_mpc_per_pix]
    counts_z is log10 z-scored by the dataset build and 0 wherever xray_mask
    is 0; XrayEncoder rewrites those non-detections itself and takes
    'observed' from expmap > 0, which separates the ~24% of the frame off the
    ACIS-I footprint from a genuine non-detection. xray_mosaic is not loaded:
    it is counts / (expmap * t_exp * pixel area), so it adds nothing. The
    X-ray inputs use fixed constants in every build, so they pool as stored.
    """
    N = int(resolution)
    print(f"\nLoading {len(file_paths)} file(s) at N={N}...")
    start = time.time()

    if keys is None:
        keys = [None] * len(file_paths)
    if len(keys) != len(file_paths):
        raise ValueError(f"keys has {len(keys)} entries for {len(file_paths)} files")

    # Pass 1: sizes, and the settings every file must share to be pooled.
    n_per_file, suites, metas, keys_per_file = [], [], [], []
    for p, k in zip(file_paths, keys):
        with h5py.File(p, "r") as f:
            if N not in [int(r) for r in f.attrs["resolutions"]]:
                raise ValueError(f"{p} has resolutions {list(f.attrs['resolutions'])}, not {N}")
            k = sorted(f.keys()) if k is None else sorted(str(s) for s in k)
            missing = [s for s in k if s not in f]
            if missing:
                raise KeyError(f"{p} has no samples {missing[:5]}")
            keys_per_file.append(k)
            n_per_file.append(len(k))
            suites.append(_suite_name(f, p))
            metas.append({k: f.attrs[k] for k in f.attrs.keys()})
    for key in ("map_fov_mpc", "shape_grid_size", "xray_grid_size", "max_nodes", "floor_value"):
        vals = {str(m.get(key)) for m in metas}
        if len(vals) > 1:
            raise ValueError(f"files disagree on {key}: {vals}")
    n = int(sum(n_per_file))

    with h5py.File(file_paths[0], "r") as f:
        first = f[keys_per_file[0][0]]
        shp = {k: first[k].shape for k in
               (f"images_{N}", f"density_cube_{N}", "gal_features",
                f"gal_pixel_coords_{N}", "mask", "src_cell_ptr", "xray_counts")}
    G = shp["xray_counts"][0]

    images = np.zeros((n, *shp[f"images_{N}"]), dtype=np.float32)
    cubes = np.zeros((n, *shp[f"density_cube_{N}"]), dtype=np.float32)
    gal_features = np.zeros((n, *shp["gal_features"]), dtype=np.float32)
    gal_pixel_coords = np.zeros((n, *shp[f"gal_pixel_coords_{N}"]), dtype=np.float32)
    mask = np.zeros((n, *shp["mask"]), dtype=np.float32)
    globals_phys = np.zeros((n, 4), dtype=np.float32)
    src_features, src_cell_id = [], []
    src_cell_ptr = np.zeros((n, *shp["src_cell_ptr"]), dtype=np.int64)
    n_sources = np.zeros((n,), dtype=np.int64)
    xray_maps = np.zeros((n, G, G, 3), dtype=np.float32)
    xray_scalars = np.zeros((n, 3), dtype=np.float32)
    suite, simulation, sample_key = [], [], []
    cluster_index = np.zeros((n,), dtype=np.int32)
    projection_index = np.zeros((n,), dtype=np.int32)

    # Pass 2: fill.
    i = 0
    for p, s_name, meta, k in zip(file_paths, suites, metas, keys_per_file):
        mean_N = float(meta[f"cube_log10_mean_{N}"])
        std_N = float(meta[f"cube_log10_std_{N}"])
        floor_file = float(meta["floor_value"])
        with h5py.File(p, "r") as f:
            for sid in k:
                g = f[sid]
                images[i] = g[f"images_{N}"][:]
                z = g[f"density_cube_{N}"][:]
                empty = z <= floor_file + 1e-6
                z = (z * std_N + (mean_N - CUBE_LOG10_MEAN)) / CUBE_LOG10_STD
                z = np.maximum(z, CUBE_FLOOR_VALUE)
                z[empty] = CUBE_FLOOR_VALUE
                cubes[i] = z
                gal_features[i] = g["gal_features"][:]
                gal_pixel_coords[i] = g[f"gal_pixel_coords_{N}"][:]
                mask[i] = g["mask"][:]
                globals_phys[i, 0] = g["cube_mass_log10_msun"][()]
                globals_phys[i, 1:] = g["axis_lengths_mpc"][:]
                src_features.append(
                    np.ascontiguousarray(g["src_features"][:, :N_SRC_FEATURES]))
                src_cell_id.append(g["src_cell_id"][:])
                src_cell_ptr[i] = g["src_cell_ptr"][:]
                n_sources[i] = g.attrs["n_sources"]
                xray_maps[i, ..., 0] = g["xray_counts"][:]
                xray_maps[i, ..., 1] = g["xray_mask"][:]
                xray_maps[i, ..., 2] = g["xray_expmap"][:]
                # Raw, not rescaled: XrayEncoder does its own O(1) scaling, and
                # it reconstructs the pixel scale as 10**scalars[:, 2].
                xray_scalars[i] = (
                    g.attrs["z_lens"],
                    np.log10(g.attrs["xray_t_exp_ks"]),
                    np.log10(g.attrs["xray_mpc_per_pix"]),
                )
                suite.append(s_name)
                simulation.append(str(g.attrs["simulation"]))
                sample_key.append(sid)
                cluster_index[i] = g.attrs["cluster_index"]
                projection_index[i] = g.attrs["projection_index"]
                i += 1
        print(f"  {s_name}: {os.path.basename(p)}, {n_per_file[suites.index(s_name)]} samples "
              f"(cube stats {mean_N:.3f}/{std_N:.3f} -> nominal "
              f"{CUBE_LOG10_MEAN}/{CUBE_LOG10_STD})", flush=True)

    metadata = dict(metas[0])
    metadata.update(resolution=N, files=list(file_paths), suites=list(suites),
                    cube_log10_mean=CUBE_LOG10_MEAN, cube_log10_std=CUBE_LOG10_STD,
                    cube_floor_value=CUBE_FLOOR_VALUE)

    size_b = (images.nbytes + cubes.nbytes + gal_features.nbytes + gal_pixel_coords.nbytes
              + mask.nbytes + src_cell_ptr.nbytes + xray_maps.nbytes
              + sum(a.nbytes for a in src_features) + sum(a.nbytes for a in src_cell_id))
    det = xray_maps[..., 1].mean(axis=(1, 2))
    obs = (xray_maps[..., 2] > 0.0).mean(axis=(1, 2))
    print(f"✓ Loaded {n} samples in {time.time() - start:.1f}s ({size_b / 1e9:.2f} GB)")
    print(f"  sources/sample: min={n_sources.min()}, median={int(np.median(n_sources))}, "
          f"max={n_sources.max()}; shape grid {int(metadata['shape_grid_size'])}^2")
    print(f"  xray {G}x{G}: detected frac median={np.median(det):.3f}, observed frac "
          f"median={np.median(obs):.3f}; t_exp {10 ** xray_scalars[:, 1].min():.0f}-"
          f"{10 ** xray_scalars[:, 1].max():.0f} ks")
    print(f"  cubes: floor-voxel frac {np.mean(cubes <= CUBE_FLOOR_VALUE):.4f}, "
          f"non-floor z range {cubes[cubes > CUBE_FLOOR_VALUE].min():.2f}.."
          f"{cubes.max():.2f}")

    return dict(
        images=images,
        cubes=cubes,
        gal_features=gal_features,
        gal_pixel_coords=gal_pixel_coords,
        mask=mask,
        globals_phys=globals_phys,
        src_features=src_features,
        src_cell_id=src_cell_id,
        src_cell_ptr=src_cell_ptr,
        n_sources=n_sources,
        xray_maps=xray_maps,
        xray_scalars=xray_scalars,
        suite=np.array(suite),
        simulation=np.array(simulation),
        sample_key=np.array(sample_key),
        cluster_index=cluster_index,
        projection_index=projection_index,
        metadata=metadata,
    )


def split_by_cluster(data: Dict[str, np.ndarray], val_frac: float = 0.2,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train/val sample indices, split by cluster rather than by sample: every
    projection of a cluster, in every run of its suite, lands on the same side.
    The same halo index recurs across a suite's runs (shared initial
    conditions), so the group key is (suite, cluster_index). Each suite is
    split val_frac separately, so both are represented in both splits.
    """
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(data["suite"]), dtype=bool)
    for s in sorted(set(data["suite"])):
        in_s = data["suite"] == s
        clusters = np.unique(data["cluster_index"][in_s])
        n_val = int(round(val_frac * len(clusters)))
        val_clusters = rng.choice(clusters, size=n_val, replace=False)
        is_val |= in_s & np.isin(data["cluster_index"], val_clusters)
        print(f"  split {s}: {len(clusters) - n_val} train / {n_val} val clusters "
              f"({np.sum(in_s & ~is_val)} / {np.sum(in_s & is_val)} samples)")
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def make_src_batch(
    data: Dict[str, np.ndarray],
    bidx: np.ndarray,
    n_src_pad: int,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble the ragged source point clouds of one batch into fixed-shape
    arrays for jit:
      src_features (B, n_src_pad, 7) zero-padded
      src_cell_id  (B, n_src_pad)    padded rows get the dummy cell S*S,
                                     which SourceShapeEncoder drops
      src_log_count(B, S, S, 1)      log1p(sources per cell), from cell_ptr
    """
    B = len(bidx)
    S = grid_size
    feats = np.zeros((B, n_src_pad, 7), dtype=np.float32)
    cids = np.full((B, n_src_pad), S * S, dtype=np.int32)
    logc = np.zeros((B, S, S, 1), dtype=np.float32)
    for j, i in enumerate(bidx):
        f = data["src_features"][i]
        n = f.shape[0]
        if n > n_src_pad:
            raise ValueError(
                f"sample {i} has {n} sources > n_src_pad={n_src_pad}; "
                "recompute n_src_pad as the max over all splits"
            )
        feats[j, :n] = f
        cids[j, :n] = data["src_cell_id"][i]
        counts = np.diff(data["src_cell_ptr"][i]).reshape(S, S)
        logc[j, ..., 0] = np.log1p(counts)
    return feats, cids, logc


def data_loader(
    data: Dict[str, np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
    n_src_pad: Optional[int] = None,
    grid_size: Optional[int] = None,
    indices: Optional[np.ndarray] = None,
) -> Iterator[Tuple[jnp.ndarray, ...]]:
    """
    Batches over `indices` (a split from split_by_cluster; all samples when
    None) of the pooled data from load_datasets.
    """
    idx = np.arange(data["images"].shape[0]) if indices is None else np.array(indices)
    n = idx.size
    if shuffle:
        rng.shuffle(idx)

    # Padded source-cloud size / shape grid; pass shared values across splits
    # so train and eval batches have identical shapes (single jit compile).
    if n_src_pad is None:
        n_src_pad = int(np.max(data["n_sources"]))
    if grid_size is None:
        grid_size = int(data["metadata"]["shape_grid_size"])

    for i in range(0, n, batch_size):
        bidx = idx[i:i + batch_size]
        if len(bidx) == 0:
            continue

        images = data["images"][bidx]                    # (B,C,H,W)
        images = np.transpose(images, (0, 2, 3, 1))     # (B,H,W,C)

        cubes = data["cubes"][bidx]                      # (B,Z,Y,X)
        cubes = cubes[..., None]                         # (B,Z,Y,X,1)

        gal_features = data["gal_features"][bidx]        # (B,N,4)
        gal_pixel_coords = data["gal_pixel_coords"][bidx]# (B,N,2)
        mask = data["mask"][bidx]                        # (B,N)

        src_features, src_cell_id, src_log_count = make_src_batch(
            data, bidx, n_src_pad, grid_size)

        # Already batch-ready, so no assembly helper is needed. A file without
        # X-ray yields None, which jax treats as an empty pytree node: it
        # passes through jit and reaches XrayEncoder's `is not None` guard,
        # at the cost of one extra compile for that batch shape.
        if "xray_maps" in data:
            xray_maps = jnp.asarray(data["xray_maps"][bidx])       # (B,G,G,3)
            xray_scalars = jnp.asarray(data["xray_scalars"][bidx])  # (B,3)
        else:
            xray_maps = None
            xray_scalars = None

        yield (
            jnp.asarray(images),
            jnp.asarray(cubes),
            jnp.asarray(gal_features),
            jnp.asarray(gal_pixel_coords),
            jnp.asarray(mask),
            jnp.asarray(src_features),
            jnp.asarray(src_cell_id),
            jnp.asarray(src_log_count),
            xray_maps,
            xray_scalars,
        )


def infinite_data_loader(data, batch_size, rng, shuffle=True,
                         n_src_pad=None, grid_size=None, indices=None):
    while True:
        yield from data_loader(data, batch_size=batch_size, rng=rng, shuffle=shuffle,
                               n_src_pad=n_src_pad, grid_size=grid_size,
                               indices=indices)


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    betas = jnp.linspace(beta_start, beta_end, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0)
    return betas, alphas, alpha_bars


def q_sample(x0, t, noise, alpha_bars):
    a_bar = alpha_bars[t]
    while a_bar.ndim < x0.ndim:
        a_bar = a_bar[..., None]
    return jnp.sqrt(a_bar) * x0 + jnp.sqrt(1.0 - a_bar) * noise


def create_train_state(model, rng_key, learning_rate, grad_clipping, example_batch):
    (images_ex, cubes_ex, gal_features_ex, gal_pixel_coords_ex, mask_ex,
     src_features_ex, src_cell_id_ex, src_log_count_ex,
     xray_maps_ex, xray_scalars_ex) = example_batch
    B = cubes_ex.shape[0]
    t_ex = jnp.zeros((B,), dtype=jnp.int32)

    params = model.init(
        rng_key,
        noisy_cube=cubes_ex,
        timesteps=t_ex,
        cond_images=images_ex,
        gal_features=gal_features_ex,
        gal_pixel_coords=gal_pixel_coords_ex,
        gal_mask=mask_ex,
        src_features=src_features_ex,
        src_cell_id=src_cell_id_ex,
        src_log_count=src_log_count_ex,
        xray_maps=xray_maps_ex,
        xray_scalars=xray_scalars_ex,
    )["params"]

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clipping),
        optax.adam(learning_rate),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def diffusion_loss(
    params,
    apply_fn,
    images,
    cubes,
    gal_features,
    gal_pixel_coords,
    mask,
    src_features,
    src_cell_id,
    src_log_count,
    xray_maps,
    xray_scalars,
    timesteps,
    noise,
    alpha_bars,
    floor_value=-5.0,
    bg_weight=0.02,
    fg_weight=1.0,
):
    noisy = q_sample(cubes, timesteps, noise, alpha_bars)
    pred_noise = apply_fn(
        {"params": params},
        noisy_cube=noisy,
        timesteps=timesteps,
        cond_images=images,
        gal_features=gal_features,
        gal_pixel_coords=gal_pixel_coords,
        gal_mask=mask,
        src_features=src_features,
        src_cell_id=src_cell_id,
        src_log_count=src_log_count,
        xray_maps=xray_maps,
        xray_scalars=xray_scalars,
    )
    #weights = jnp.where(cubes <= floor_value + 1e-6, bg_weight, fg_weight)
    #sqerr = (pred_noise - noise) ** 2
    return jnp.mean((pred_noise - noise) ** 2)
    #return jnp.sum(weights * sqerr) / jnp.sum(weights)


@jax.jit
def train_step(state, images, cubes, gal_features, gal_pixel_coords, mask,
               src_features, src_cell_id, src_log_count,
               xray_maps, xray_scalars,
               rng_key, alpha_bars, num_timesteps):
    rng_t, rng_n = jax.random.split(rng_key, 2)
    B = cubes.shape[0]
    t = jax.random.randint(rng_t, shape=(B,), minval=0, maxval=num_timesteps)
    noise = jax.random.normal(rng_n, shape=cubes.shape)

    def loss_fn(params):
        return diffusion_loss(
            params,
            state.apply_fn,
            images,
            cubes,
            gal_features,
            gal_pixel_coords,
            mask,
            src_features,
            src_cell_id,
            src_log_count,
            xray_maps,
            xray_scalars,
            t,
            noise,
            alpha_bars,
        )

    grads = jax.grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    loss = loss_fn(new_state.params)
    return new_state, loss


@jax.jit
def eval_step(state, images, cubes, gal_features, gal_pixel_coords, mask,
              src_features, src_cell_id, src_log_count,
              xray_maps, xray_scalars,
              rng_key, alpha_bars, num_timesteps):
    rng_t, rng_n = jax.random.split(rng_key, 2)
    B = cubes.shape[0]
    t = jax.random.randint(rng_t, shape=(B,), minval=0, maxval=num_timesteps)
    noise = jax.random.normal(rng_n, shape=cubes.shape)

    return diffusion_loss(
        state.params,
        state.apply_fn,
        images,
        cubes,
        gal_features,
        gal_pixel_coords,
        mask,
        src_features,
        src_cell_id,
        src_log_count,
        xray_maps,
        xray_scalars,
        t,
        noise,
        alpha_bars,
    )


def sample_ddpm(
    model_apply,
    params,
    cond_images,
    gal_features,
    gal_pixel_coords,
    mask,
    src_features,
    src_cell_id,
    src_log_count,
    sample_shape,
    rng_key,
    betas,
    alphas,
    alpha_bars,
    xray_maps=None,
    xray_scalars=None,
):
    """
    cond_images:      (B,H,W,C)
    gal_features:     (B,N,4)
    gal_pixel_coords: (B,N,2)
    mask:             (B,N)
    src_features:     (B,Ns,7)
    src_cell_id:      (B,Ns)
    src_log_count:    (B,S,S,1)
    sample_shape:     (B,Z,Y,X,1)
    xray_maps:        (B,G,G,3) or None
    xray_scalars:     (B,3) or None

    Keyword-only at the end so existing positional callers keep working; pass
    both if the checkpoint was trained with the X-ray branch.
    """
    x = jax.random.normal(rng_key, shape=sample_shape)

    T = betas.shape[0]
    for step in reversed(range(T)):
        t = jnp.full((sample_shape[0],), step, dtype=jnp.int32)

        pred_noise = model_apply(
            {"params": params},
            noisy_cube=x,
            timesteps=t,
            cond_images=cond_images,
            gal_features=gal_features,
            gal_pixel_coords=gal_pixel_coords,
            gal_mask=mask,
            src_features=src_features,
            src_cell_id=src_cell_id,
            src_log_count=src_log_count,
            xray_maps=xray_maps,
            xray_scalars=xray_scalars,
        )

        beta_t = betas[step]
        alpha_t = alphas[step]
        alpha_bar_t = alpha_bars[step]

        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)

        mean = coef1 * (x - coef2 * pred_noise)

        if step > 0:
            rng_key, subkey = jax.random.split(rng_key)
            z = jax.random.normal(subkey, shape=x.shape)
            sigma = jnp.sqrt(beta_t)
            x = mean + sigma * z
        else:
            x = mean

    return x


def train_model(
    data: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model,
    batch_size: int = 2,
    num_train_steps: int = 100_000,
    eval_every: int = 250,
    log_every: int = 50,
    num_eval_batches: Optional[int] = 50,
    learning_rate=1e-4,
    grad_clipping: float = 1.0,
    num_diffusion_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    wandb_project: str = "conditional-3d-diffusion",
    wandb_run_name: Optional[str] = None,
    wandb_notes: str = "",
    cfg_dict: Optional[dict] = None,
    checkpoint_dir: str = "./runtime_checkpoints",
    checkpoint_prefix: str = "cond_diffusion_runtime",
    max_runtime_hours: float = 7.8,
    runtime_buffer_minutes: float = 10.0,
):
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)

    train_rng = np.random.default_rng(42)
    # One validation stream per suite, so each suite's loss is reported on its
    # own; val_loss is their sample-weighted mean.
    val_suites = sorted(set(data["suite"][val_idx]))
    val_idx_by_suite = {s: val_idx[data["suite"][val_idx] == s] for s in val_suites}
    val_rngs = {s: np.random.default_rng(123 + k) for k, s in enumerate(val_suites)}

    os.makedirs(checkpoint_dir, exist_ok=True)

    betas, alphas, alpha_bars = make_beta_schedule(
        num_diffusion_steps, beta_start=beta_start, beta_end=beta_end
    )

    diffusion_cfg = dict(
        betas=np.array(betas),
        alphas=np.array(alphas),
        alpha_bars=np.array(alpha_bars),
        num_diffusion_steps=num_diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    )

    # Shared padded source-cloud size across train and eval so every batch has
    # identical shapes (one jit compile of each step function).
    n_src_pad = int(np.max(data["n_sources"]))
    grid_size = int(data["metadata"]["shape_grid_size"])
    print(f"source clouds padded to n_src_pad={n_src_pad}, shape grid {grid_size}x{grid_size}")
    print(f"train {len(train_idx)} samples; val " + ", ".join(
        f"{s} {len(v)}" for s, v in val_idx_by_suite.items()))

    train_stream = infinite_data_loader(data, batch_size, rng=train_rng, shuffle=True,
                                        n_src_pad=n_src_pad, grid_size=grid_size,
                                        indices=train_idx)
    example_batch = next(
        data_loader(
            data,
            batch_size=min(batch_size, 2),
            rng=train_rng,
            shuffle=False,
            n_src_pad=n_src_pad,
            grid_size=grid_size,
            indices=train_idx,
        )
    )

    state = create_train_state(model, init_key, learning_rate, grad_clipping, example_batch)

    run = wandb.init(
        entity="erichabjan-northeastern-university",
        project=wandb_project,
        name=wandb_run_name,
        config=dict(
            learning_rate=str(learning_rate),
            grad_clipping=float(grad_clipping),
            batch_size=int(batch_size),
            num_diffusion_steps=int(num_diffusion_steps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            notes=wandb_notes,
            **({} if cfg_dict is None else cfg_dict),
        ),
    )

    def save_runtime_checkpoint(state, step, train_losses, test_losses, reason):
        """
        Saves:
          1. full TrainState as flax bytes
          2. metadata / losses / diffusion schedule as pickle
        """
        state_path = os.path.join(
            checkpoint_dir,
            f"{checkpoint_prefix}_state_step{step}.msgpack"
        )
        meta_path = os.path.join(
            checkpoint_dir,
            f"{checkpoint_prefix}_meta_step{step}.pkl"
        )

        with open(state_path, "wb") as f:
            f.write(serialization.to_bytes(state))

        meta = dict(
            step=int(step),
            train_losses=np.asarray(train_losses, dtype=np.float32),
            test_losses=np.asarray(test_losses, dtype=np.float32),
            val_losses_by_suite={s: np.asarray(v, dtype=np.float32)
                                 for s, v in val_losses_by_suite.items()},
            diffusion_cfg=diffusion_cfg,
            # What a prediction means: log10 rho = z * std + mean.
            cube_log10_mean=float(data["metadata"]["cube_log10_mean"]),
            cube_log10_std=float(data["metadata"]["cube_log10_std"]),
            resolution=int(data["metadata"]["resolution"]),
            train_idx=np.asarray(train_idx),
            val_idx=np.asarray(val_idx),
            reason=str(reason),
            wandb_run_id=None if run is None else run.id,
        )

        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        print(f"\nSaved runtime checkpoint:")
        print(f"  state: {state_path}")
        print(f"  meta : {meta_path}")
        print(f"  reason: {reason}")

    def eval_loop():
        """
        Validation loss per suite (num_eval_batches batches each, or all of
        them when None), and the sample-weighted mean over suites.
        """
        per_suite, n_seen = {}, {}
        for s_name in val_suites:
            total, count, n_s = 0.0, 0, 0
            it = data_loader(data, batch_size=batch_size, rng=val_rngs[s_name], shuffle=True,
                             n_src_pad=n_src_pad, grid_size=grid_size,
                             indices=val_idx_by_suite[s_name])
            for k, batch in enumerate(it):
                if num_eval_batches is not None and k >= num_eval_batches:
                    break
                loss = eval_step(
                    state, *batch,
                    jax.random.PRNGKey(1000 + k), alpha_bars, num_diffusion_steps
                )
                b = int(batch[1].shape[0])
                total += float(loss) * b
                n_s += b
            per_suite[s_name] = total / max(n_s, 1)
            n_seen[s_name] = n_s
        overall = sum(per_suite[s] * n_seen[s] for s in val_suites) / max(sum(n_seen.values()), 1)
        return overall, per_suite

    def log_val(step, label="val_loss"):
        loss_val, per_suite = eval_loop()
        test_losses.append(loss_val)
        for s_name, v in per_suite.items():
            val_losses_by_suite[s_name].append(v)
        print(f"Step {step} | {label}: {loss_val:.6f} | " + " | ".join(
            f"{s_name}: {v:.6f}" for s_name, v in per_suite.items()))
        run.log({"val_loss": loss_val,
                 **{f"val_loss_{s_name}": v for s_name, v in per_suite.items()},
                 "step": step}, step=step)

    train_losses = []
    test_losses = []
    val_losses_by_suite = {s: [] for s in val_suites}

    start_time = time.time()
    soft_limit_seconds = max_runtime_hours * 3600.0
    buffer_seconds = runtime_buffer_minutes * 60.0

    for step in range(1, num_train_steps + 1):
        batch = next(train_stream)
        rng_key, step_key = jax.random.split(rng_key)

        state, loss_train = train_step(
            state,
            *batch,
            step_key,
            alpha_bars,
            num_diffusion_steps,
        )
        loss_train.block_until_ready()
        loss_train = float(loss_train)

        train_losses.append(loss_train)
        run.log({"train_loss": loss_train, "step": step}, step=step)

        if step % log_every == 0:
            elapsed_hours = (time.time() - start_time) / 3600.0
            print(f"Step {step} | train_loss: {loss_train:.6f} | elapsed: {elapsed_hours:.2f} hr")

        if step % eval_every == 0:
            log_val(step)

        # Check walltime after the step has fully completed
        elapsed_seconds = time.time() - start_time
        time_remaining = soft_limit_seconds - elapsed_seconds

        if time_remaining <= buffer_seconds:
            print(
                f"\nApproaching runtime limit: "
                f"elapsed={elapsed_seconds/3600.0:.2f} hr, "
                f"remaining={time_remaining/60.0:.2f} min"
            )

            # Optional: run one last validation before saving if you want
            # and if this step was not already an eval step.
            if step % eval_every != 0:
                log_val(step, label="final pre-exit val_loss")

            save_runtime_checkpoint(
                state=state,
                step=step,
                train_losses=train_losses,
                test_losses=test_losses,
                reason="approaching_walltime",
            )

            run.finish()
            return state, np.asarray(train_losses), np.asarray(test_losses), diffusion_cfg

    # Normal completion
    save_runtime_checkpoint(
        state=state,
        step=num_train_steps,
        train_losses=train_losses,
        test_losses=test_losses,
        reason="finished_training",
    )

    run.finish()
    return state, np.asarray(train_losses), np.asarray(test_losses), diffusion_cfg