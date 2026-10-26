#!/usr/bin/env python3
"""
Posterior sampler for the conditional diffusion model on a held-out
shape_dynamics test suite (TNG300 by default: models are trained on BAHAMAS
+ FLAMINGO, see run_conditional_diffusion.py).

One run of this script samples one simulation of the suite at one cube
resolution N, i.e. one model:

  - selects one projection (default 0) of every cluster in --simulation and
    loads only those samples (load_datasets(keys=...))
  - draws --n-draws independent DDPM samples per cluster
  - packs (cluster, draw) slots into full GPU batches, crossing cluster
    boundaries so every launch is full
  - per launch, the conditioning branches (source shapes, X-ray, images,
    galaxy tokens) run ONCE (ConditionalUNet3D encode_only=True); the
    1000-step reverse chain then reuses them inside one jitted fori_loop
  - galaxy arrays are cut from max_nodes to the most real galaxies among the
    selected samples: padded galaxies are masked out of the attention, so this
    is exact, and it is what bounds memory and time at N = 32
  - every draw's noise comes from its own key, fold_in(seed, sample, draw),
    so a draw does not depend on batch packing or on resuming
  - writes one npz per cluster the moment its draws are complete, and skips
    clusters whose npz already exists (crash/walltime-safe resume)

Cubes, sampled and true, are in the nominal standardization of
train_conditional_diffusion: log10(rho / Msun Mpc^-3) = z * std + mean, with
mean/std saved in each npz; empty voxels are at the floor value.

Run: run_cond_diff_sampler.slurm (A100), one array task per (N, simulation).
"""
import os
import sys
import time
import argparse
import pickle
import h5py
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from conditional_diffusion_3d_model import ConditionalUNet3D, DiffusionModelConfig
from train_conditional_diffusion import (load_datasets, make_src_batch, CUBE_LOG10_MEAN,
                                         CUBE_LOG10_STD, CUBE_FLOOR_VALUE)

# ============================================================
# Config
# ============================================================
DATA_PATH = "/projects/mccleary_group/habjan.e/TNG/Data/shape_dynamics/tng300_dataset.hdf5"
MODEL_DIR = "/home/habjan.e/TNG/cluster_deprojection/conditional_diffusion_model/conditional_diffusion_models"
# One subdirectory per model under this.
OUT_ROOT = "/projects/mccleary_group/habjan.e/TNG/Data/shape_dynamics/cd_posterior"

# Model files are cond_diffusion_{params,schedule}{MODEL_TAG}_N{N}.pkl.
MODEL_TAG = "_shape_dyn_xray_bahamas_flamingo"

N_POSTERIOR_SAMPLES = 50     # independent posterior draws per cluster
BASE_SAMPLE_SEED = 0         # PRNG base for the diffusion noise draws

# Max (example, draw) slots per GPU launch. The ceiling is the galaxy
# cross-attention: its logits are (B, heads, N^3 queries, n_gal) float32 at
# every step, ~0.55 GB per slot at N = 32 with ~1000 galaxies. Unless
# --batch-size is given, the batch is the largest that keeps those logits
# under ATTN_LOGITS_BUDGET_GB, capped at MAX_BATCH_SIZE. 6 GB leaves room on
# a 40 GB A100 for the softmax copy and the U-Net.
MAX_BATCH_SIZE = 64
ATTN_LOGITS_BUDGET_GB = 6.0


def parse_args():
    p = argparse.ArgumentParser(description="Posterior sampling of one simulation at one resolution.")
    p.add_argument("--resolution", type=int, required=True, choices=(16, 32, 64, 128),
                   help="N of the model (and of the N^3 cube / N x N images)")
    p.add_argument("--simulation", required=True,
                   help="simulation attr of the samples to use, e.g. TNG300-1")
    p.add_argument("--projection", type=int, default=0,
                   help="projection_index sampled for every cluster")
    p.add_argument("--n-draws", type=int, default=N_POSTERIOR_SAMPLES)
    p.add_argument("--batch-size", type=int, default=None,
                   help="slots per launch; default from ATTN_LOGITS_BUDGET_GB")
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--seed", type=int, default=BASE_SAMPLE_SEED)
    return p.parse_args()


# ============================================================
# Sampling machinery
# ============================================================

def make_sampler(model, betas, alphas, alpha_bars, base_seed):
    """
    Returns sampler(params, cond, sample_ids, draw_ids). One call encodes the
    conditioning once, then runs the FULL reverse chain for the batch inside
    a single jitted fori_loop (one GPU dispatch per launch). The DDPM update
    mirrors train_conditional_diffusion.sample_ddpm exactly. Slot b's noise
    comes from fold_in(fold_in(base, sample_ids[b]), draw_ids[b]). Compiled
    once per (batch shape, cube shape).
    """
    T = int(betas.shape[0])
    base_key = jax.random.PRNGKey(base_seed)
    _cache = {}

    def sampler(params, cond, sample_ids, draw_ids):
        cond = dict(cond)
        cube_shape = cond.pop("cube_shape")

        def fn(p, c, sids, dids):
            B = c["cond_images"].shape[0]
            shape = (*cube_shape, 1)

            cond_cache = model.apply(
                {"params": p},
                noisy_cube=jnp.zeros((B, *shape), dtype=jnp.float32),
                timesteps=jnp.zeros((B,), dtype=jnp.int32),
                encode_only=True, **c)

            keys = jax.vmap(lambda s, d: jax.random.fold_in(
                jax.random.fold_in(base_key, s), d))(sids, dids)
            keys = jax.vmap(jax.random.split)(keys)                  # (B,2,2)
            x0 = jax.vmap(lambda k: jax.random.normal(k, shape))(keys[:, 0])

            def body(i, carry):
                x, key = carry
                step = T - 1 - i
                t = jnp.full((B,), step, dtype=jnp.int32)
                pred_noise = model.apply({"params": p}, noisy_cube=x,
                                         timesteps=t, cond_cache=cond_cache)
                beta_t = betas[step]
                alpha_t = alphas[step]
                alpha_bar_t = alpha_bars[step]
                mean = (x - beta_t / jnp.sqrt(1.0 - alpha_bar_t) * pred_noise) \
                    / jnp.sqrt(alpha_t)
                key = jax.vmap(jax.random.split)(key)                # (B,2,2)
                z = jax.vmap(lambda k: jax.random.normal(k, shape))(key[:, 1])
                x = jnp.where(step > 0, mean + jnp.sqrt(beta_t) * z, mean)
                return (x, key[:, 0])

            x, _ = jax.lax.fori_loop(0, T, body, (x0, keys[:, 1]))
            return x

        if cube_shape not in _cache:
            _cache[cube_shape] = jax.jit(fn)
        return _cache[cube_shape](params, cond, jnp.asarray(sample_ids, dtype=jnp.int32),
                                  jnp.asarray(draw_ids, dtype=jnp.int32))

    return sampler


def build_launch_cond(data, idx, n_src_pad, n_gal_pad, grid_size):
    """
    Assemble the conditioning for one launch. idx (length B) indexes the
    loaded samples and may repeat: repeated entries are independent posterior
    draws of that sample. Galaxy arrays are cut to their first n_gal_pad rows
    (real galaxies come first in every sample, padding after). Returns the
    conditioning kwargs for model.apply, plus the static cube shape.
    """
    images = data["images"][idx]                         # (B,C,H,W)
    images = np.transpose(images, (0, 2, 3, 1))          # (B,H,W,C)
    src_features, src_cell_id, src_log_count = make_src_batch(
        data, idx, n_src_pad, grid_size)
    return dict(
        cond_images=jnp.asarray(images, dtype=jnp.float32),
        gal_features=jnp.asarray(data["gal_features"][idx, :n_gal_pad]),
        gal_pixel_coords=jnp.asarray(data["gal_pixel_coords"][idx, :n_gal_pad]),
        gal_mask=jnp.asarray(data["mask"][idx, :n_gal_pad]),
        src_features=jnp.asarray(src_features),
        src_cell_id=jnp.asarray(src_cell_id),
        src_log_count=jnp.asarray(src_log_count),
        xray_maps=jnp.asarray(data["xray_maps"][idx]),
        xray_scalars=jnp.asarray(data["xray_scalars"][idx]),
        cube_shape=tuple(int(s) for s in data["cubes"].shape[1:]),
    )


def example_out_path(out_dir, data, i):
    return os.path.join(
        out_dir,
        f"posterior_{data['simulation'][i]}_cl{int(data['cluster_index'][i]):03d}"
        f"_proj{int(data['projection_index'][i])}.npz")


def save_example(out_dir, data, i, cubes, model_name, seed):
    np.savez_compressed(
        example_out_path(out_dir, data, i),
        sampled_cubes=cubes.astype(np.float32),                 # (n_draws,Z,Y,X)
        true_cube=data["cubes"][i].astype(np.float32),          # (Z,Y,X)
        conditioning_images=data["images"][i].astype(np.float32),  # (C,H,W)
        cube_log10_mean=np.float32(CUBE_LOG10_MEAN),
        cube_log10_std=np.float32(CUBE_LOG10_STD),
        cube_floor_value=np.float32(CUBE_FLOOR_VALUE),
        resolution=np.int32(data["cubes"].shape[1]),
        sample_key=str(data["sample_key"][i]),
        suite=str(data["suite"][i]),
        simulation=str(data["simulation"][i]),
        cluster_index=np.int32(data["cluster_index"][i]),
        projection_index=np.int32(data["projection_index"][i]),
        z_lens=np.float32(data["xray_scalars"][i, 0]),
        cube_mass_log10_msun=np.float32(data["globals_phys"][i, 0]),
        axis_lengths_mpc=data["globals_phys"][i, 1:].astype(np.float32),
        n_sources=np.int64(data["n_sources"][i]),
        n_galaxies=np.int32(data["mask"][i].sum()),
        n_posterior_samples=np.int32(cubes.shape[0]),
        base_sample_seed=np.int32(seed),
        model=model_name,
    )


def run_sampling(model, params, data, betas, alphas, alpha_bars,
                 indices, n_draws, batch_size, n_src_pad, n_gal_pad, grid_size,
                 out_dir, model_name, base_seed=0):
    """
    Draw n_draws posterior samples for each entry of indices (into the loaded
    data), packing (sample, draw) slots into full batches. Saves one cube
    stack per sample; samples with an existing npz are skipped.
    """
    os.makedirs(out_dir, exist_ok=True)
    sampler = make_sampler(model, betas, alphas, alpha_bars, base_seed)

    todo = [i for i in indices
            if not os.path.exists(example_out_path(out_dir, data, i))]
    skipped = len(indices) - len(todo)
    if skipped:
        print(f"resume: {skipped} clusters already sampled, {len(todo)} to go")
    if not todo:
        return

    # Flat work list of (sample, draw) slots, samples in order. The noise key
    # uses the sample's group name in the file, so it is stable across runs.
    slots = np.repeat(np.asarray(todo, dtype=np.int64), n_draws)
    draws = np.tile(np.arange(n_draws, dtype=np.int64), len(todo))
    file_ids = np.array([int(k) for k in data["sample_key"]], dtype=np.int64)
    n_launches = int(np.ceil(slots.size / batch_size))
    print(f"{len(todo)} clusters x {n_draws} draws = {slots.size} slots "
          f"-> {n_launches} launches of <= {batch_size}")

    buffers = {int(i): [] for i in todo}
    done_ptr = 0  # samples are completed in order
    cube_zyx = tuple(int(s) for s in data["cubes"].shape[1:])
    t_start = time.time()

    for li in range(n_launches):
        launch_idx = slots[li * batch_size:(li + 1) * batch_size]
        launch_draw = draws[li * batch_size:(li + 1) * batch_size]
        t0 = time.time()
        cond = build_launch_cond(data, launch_idx, n_src_pad, n_gal_pad, grid_size)
        sampled = sampler(params, cond, file_ids[launch_idx], launch_draw)
        sampled = np.asarray(sampled[..., 0], dtype=np.float32)  # (B,Z,Y,X)
        dt = time.time() - t0

        for i, cube in zip(launch_idx, sampled):
            buffers[int(i)].append(cube)

        # Flush every sample whose draws are complete.
        while done_ptr < len(todo) and len(buffers[int(todo[done_ptr])]) >= n_draws:
            i = int(todo[done_ptr])
            cubes = np.stack(buffers.pop(i), axis=0)
            assert cubes.shape == (n_draws, *cube_zyx)
            save_example(out_dir, data, i, cubes, model_name, base_seed)
            done_ptr += 1

        print(f"launch {li + 1}/{n_launches}: {launch_idx.size} draws in "
              f"{dt:.1f}s ({dt / launch_idx.size:.2f} s/draw); "
              f"{done_ptr}/{len(todo)} clusters saved; "
              f"elapsed {(time.time() - t_start) / 3600:.2f} hr", flush=True)

    assert done_ptr == len(todo)
    print(f"done: {len(todo)} clusters written to {out_dir}")


# ============================================================
# Main
# ============================================================

def select_keys(path, simulation, projection):
    """Group names of the samples of one simulation at one projection, one per cluster."""
    keys, clusters = [], []
    with h5py.File(path, "r") as f:
        for sid in sorted(f.keys()):
            a = f[sid].attrs
            if str(a["simulation"]) == simulation and int(a["projection_index"]) == projection:
                keys.append(sid)
                clusters.append(int(a["cluster_index"]))
    if not keys:
        raise ValueError(f"no samples with simulation={simulation}, "
                         f"projection_index={projection} in {path}")
    if len(set(clusters)) != len(clusters):
        raise ValueError(f"{simulation} has several samples per cluster at projection {projection}")
    return keys


def main():
    args = parse_args()
    N = args.resolution
    suffix = f"{MODEL_TAG}_N{N}"

    keys = select_keys(args.data, args.simulation, args.projection)
    print(f"{args.simulation}: {len(keys)} clusters at projection {args.projection}")
    data = load_datasets([args.data], resolution=N, keys=[keys])
    meta = data["metadata"]
    grid_size = int(meta["shape_grid_size"])
    indices = np.arange(data["images"].shape[0])

    n_src_pad = int(data["n_sources"].max())
    n_gal_pad = int(data["mask"].sum(axis=1).max())
    print(f"padded source-cloud length n_src_pad={n_src_pad}; galaxy tokens cut "
          f"to n_gal_pad={n_gal_pad} of max_nodes={data['mask'].shape[1]}")

    param_path = os.path.join(MODEL_DIR, f"cond_diffusion_params{suffix}.pkl")
    sched_path = os.path.join(MODEL_DIR, f"cond_diffusion_schedule{suffix}.pkl")
    with open(param_path, "rb") as f:
        params = pickle.load(f)
    with open(sched_path, "rb") as f:
        diffusion_cfg = pickle.load(f)

    betas = jnp.asarray(diffusion_cfg["betas"], dtype=jnp.float32)
    alphas = jnp.asarray(diffusion_cfg["alphas"], dtype=jnp.float32)
    alpha_bars = jnp.asarray(diffusion_cfg["alpha_bars"], dtype=jnp.float32)
    print(f"loaded params from {param_path}; T={betas.shape[0]} diffusion steps")

    # Must match the training run (run_conditional_diffusion.py).
    cfg = DiffusionModelConfig(
        base_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        out_channels=1,
        galaxy_token_dim=128,
        num_attention_heads=4,
        coord_image_size=N,
        src_embed_dim=32,
        src_out_ch=32,
        shape_grid_size=grid_size,
        xray_out_ch=32,
        xray_grid_size=int(meta["xray_grid_size"]),
        xray_reproj_size=128,
        map_fov_mpc=float(meta["map_fov_mpc"]),
    )
    model = ConditionalUNet3D(cfg=cfg)

    if args.batch_size is None:
        per_slot = cfg.num_attention_heads * N ** 3 * n_gal_pad * 4
        batch_size = int(np.clip(ATTN_LOGITS_BUDGET_GB * 1e9 // per_slot, 1, MAX_BATCH_SIZE))
    else:
        batch_size = int(args.batch_size)
    print(f"batch size {batch_size} (attention logits "
          f"{batch_size * cfg.num_attention_heads * N ** 3 * n_gal_pad * 4 / 1e9:.1f} GB)")

    out_dir = os.path.join(args.out_root, f"tng300{suffix}")
    run_sampling(
        model, params, data, betas, alphas, alpha_bars,
        indices=indices,
        n_draws=args.n_draws,
        batch_size=batch_size,
        n_src_pad=n_src_pad,
        n_gal_pad=n_gal_pad,
        grid_size=grid_size,
        out_dir=out_dir,
        model_name=f"cond_diffusion_params{suffix}",
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
