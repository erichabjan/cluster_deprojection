#!/usr/bin/env python3
"""
Posterior sampler for the conditional diffusion model on the shape_dynamics
dataset. Combines the batched efficiency of sampling_scripts/
samples_examples_light.py with the seeds-per-example posterior sampling of
sampling_scripts/samples_seeds.py.

  - selects NUM_EXAMPLES random validation examples (fixed SELECTION_SEED)
  - draws N_POSTERIOR_SAMPLES independent DDPM samples per example
  - packs (example, draw) slots into full GPU batches of SAMPLING_BATCH_SIZE,
    crossing example boundaries so every launch is full
  - the whole 1000-step reverse chain runs inside one jitted fori_loop;
    a shared padded source-cloud size means it compiles exactly once
  - writes one npz per example the moment its draws are complete, and skips
    examples whose npz already exists (crash/walltime-safe resume)

Run: run_cond_diff_sampler.slurm (A100).
"""
import os
import sys
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from conditional_diffusion_3d_model import ConditionalUNet3D, DiffusionModelConfig
from train_conditional_diffusion import preload_hdf5_to_memory, make_src_batch

# ============================================================
# Config
# ============================================================
SUFFIX = "_shape_dyn_v1"

VAL_PATH = "/projects/mccleary_group/habjan.e/TNG/Data/shape_dynamics/shape_dynamics_val.h5"
MODEL_DIR = "/home/habjan.e/TNG/cluster_deprojection/conditional_diffusion_model/conditional_diffusion_models"
OUT_DIR = "/scratch/habjan.e/conditional_diffusion/posterior_sampling" #"/projects/mccleary_group/habjan.e/TNG/Data/shape_dynamics/cd_posterior"

NUM_EXAMPLES = 50            # random validation examples to sample
N_POSTERIOR_SAMPLES = 200    # independent posterior draws per example
SELECTION_SEED = 12345       # fixes which examples are selected (stable across resumes)
BASE_SAMPLE_SEED = 0         # PRNG base for the diffusion noise draws

# Number of (example, draw) slots per GPU launch. Larger = better GPU
# utilization; the ceiling is device memory used by the padded source clouds
# (~250-300 MB per slot at n_pad ~ 5e5). 64 fits a 40 GB A100; 128 needs an
# 80 GB card (add `#SBATCH --constraint=a100@80g` to the slurm file to pin
# one). Drop this if you hit OOM.
SAMPLING_BATCH_SIZE = 64

# Padded source-cloud length. None -> max n_sources over the SELECTED
# examples (recommended). Set manually to trade memory for batch size; must
# be >= the largest n_sources among the selected examples or the batch
# builder raises.
MAX_N_SOURCES = None

# Model config values; must match the training run (run_conditional_diffusion.py).
SRC_EMBED_DIM = 32
SRC_OUT_CH = 32


# ============================================================
# Sampling machinery
# ============================================================

def make_sampler(model, betas, alphas, alpha_bars):
    """
    Returns sampler(params, cond, rng_key) running the FULL reverse chain for
    one batch inside a single jitted fori_loop (one GPU dispatch per launch).
    The DDPM update mirrors train_conditional_diffusion.sample_ddpm exactly.
    Compiled once per (batch shape, cube shape); with a shared padded
    source-cloud size that means exactly once.
    """
    T = int(betas.shape[0])
    _cache = {}

    def sampler(params, cond, rng_key):
        cond = dict(cond)
        cube_shape = cond.pop("cube_shape")

        def fn(p, c, k):
            B = c["cond_images"].shape[0]
            key_init, key_loop = jax.random.split(k)
            x0 = jax.random.normal(key_init, shape=(B, *cube_shape, 1))

            def body(i, carry):
                x, key = carry
                step = T - 1 - i
                t = jnp.full((B,), step, dtype=jnp.int32)
                pred_noise = model.apply({"params": p}, noisy_cube=x,
                                         timesteps=t, **c)
                beta_t = betas[step]
                alpha_t = alphas[step]
                alpha_bar_t = alpha_bars[step]
                mean = (x - beta_t / jnp.sqrt(1.0 - alpha_bar_t) * pred_noise) \
                    / jnp.sqrt(alpha_t)
                key, sub = jax.random.split(key)
                z = jax.random.normal(sub, shape=x.shape)
                x = jnp.where(step > 0, mean + jnp.sqrt(beta_t) * z, mean)
                return (x, key)

            x, _ = jax.lax.fori_loop(0, T, body, (x0, key_loop))
            return x

        if cube_shape not in _cache:
            _cache[cube_shape] = jax.jit(fn)
        return _cache[cube_shape](params, cond, rng_key)

    return sampler


def build_launch_cond(data, val_indices, n_src_pad, grid_size):
    """
    Assemble the conditioning for one launch. val_indices (length B) may
    repeat: repeated entries are independent posterior draws of that example.
    Returns the kwargs dict for model.apply (minus noisy_cube/timesteps),
    plus the static cube shape.
    """
    images = data["images"][val_indices]                 # (B,C,H,W)
    images = np.transpose(images, (0, 2, 3, 1))          # (B,H,W,C)
    src_features, src_cell_id, src_log_count = make_src_batch(
        data, val_indices, n_src_pad, grid_size)
    return dict(
        cond_images=jnp.asarray(images, dtype=jnp.float32),
        gal_features=jnp.asarray(data["gal_features"][val_indices]),
        gal_pixel_coords=jnp.asarray(data["gal_pixel_coords"][val_indices]),
        gal_mask=jnp.asarray(data["mask"][val_indices]),
        src_features=jnp.asarray(src_features),
        src_cell_id=jnp.asarray(src_cell_id),
        src_log_count=jnp.asarray(src_log_count),
        cube_shape=tuple(int(s) for s in data["cubes"].shape[1:]),
    )


def example_out_path(out_dir, val_idx):
    return os.path.join(out_dir, f"posterior_example_{int(val_idx):04d}{SUFFIX}.npz")


def save_example(out_dir, data, val_idx, cubes):
    np.savez_compressed(
        example_out_path(out_dir, val_idx),
        test_idx=np.int32(val_idx),
        sampled_cubes=cubes.astype(np.float32),               # (n_draws,Z,Y,X)
        true_cube=data["cubes"][val_idx].astype(np.float32),  # (Z,Y,X)
        conditioning_images=data["images"][val_idx].astype(np.float32),  # (C,H,W)
        z_lens=np.float32(data["z_lens"][val_idx]),
        n_sources=np.int64(data["n_sources"][val_idx]),
        cluster_index=np.int32(data["cluster_index"][val_idx]),
        simulation=str(data["simulation"][val_idx]),
        n_posterior_samples=np.int32(cubes.shape[0]),
        base_sample_seed=np.int32(BASE_SAMPLE_SEED),
        selection_seed=np.int32(SELECTION_SEED),
    )


def run_sampling(model, params, data, betas, alphas, alpha_bars,
                 val_indices, n_draws, batch_size, n_src_pad, grid_size,
                 out_dir, base_seed=0):
    """
    Draw n_draws posterior samples for each entry of val_indices, packing
    (example, draw) slots into full batches. Saves/returns one cube stack per
    example; examples with an existing npz are skipped.
    """
    os.makedirs(out_dir, exist_ok=True)
    sampler = make_sampler(model, betas, alphas, alpha_bars)

    todo = [vi for vi in val_indices
            if not os.path.exists(example_out_path(out_dir, vi))]
    skipped = len(val_indices) - len(todo)
    if skipped:
        print(f"resume: {skipped} examples already sampled, {len(todo)} to go")
    if not todo:
        return

    # Flat work list of (example, draw) slots, examples in order.
    slots = np.repeat(np.asarray(todo, dtype=np.int64), n_draws)
    n_launches = int(np.ceil(slots.size / batch_size))
    print(f"{len(todo)} examples x {n_draws} draws = {slots.size} slots "
          f"-> {n_launches} launches of <= {batch_size}")

    buffers = {int(vi): [] for vi in todo}
    done_ptr = 0  # examples are completed in order
    cube_zyx = tuple(int(s) for s in data["cubes"].shape[1:])

    for li in range(n_launches):
        launch_idx = slots[li * batch_size:(li + 1) * batch_size]
        t0 = time.time()
        cond = build_launch_cond(data, launch_idx, n_src_pad, grid_size)
        rng_key = jax.random.PRNGKey(base_seed + li)
        sampled = sampler(params, cond, rng_key)
        sampled = np.asarray(sampled[..., 0], dtype=np.float32)  # (B,Z,Y,X)
        dt = time.time() - t0

        for vi, cube in zip(launch_idx, sampled):
            buffers[int(vi)].append(cube)

        # Flush every example whose draws are complete.
        while done_ptr < len(todo) and len(buffers[int(todo[done_ptr])]) >= n_draws:
            vi = int(todo[done_ptr])
            cubes = np.stack(buffers.pop(vi), axis=0)
            assert cubes.shape == (n_draws, *cube_zyx)
            save_example(out_dir, data, vi, cubes)
            done_ptr += 1

        print(f"launch {li + 1}/{n_launches}: {launch_idx.size} draws in "
              f"{dt:.1f}s ({dt / launch_idx.size:.2f} s/draw); "
              f"{done_ptr}/{len(todo)} examples saved", flush=True)

    assert done_ptr == len(todo)
    print(f"done: {len(todo)} examples written to {out_dir}")


# ============================================================
# Main
# ============================================================

def main():
    data = preload_hdf5_to_memory(VAL_PATH)
    n_val = data["images"].shape[0]
    grid_size = int(data["metadata"]["shape_grid_size"])

    rng = np.random.default_rng(SELECTION_SEED)
    val_indices = np.sort(rng.choice(n_val, size=NUM_EXAMPLES, replace=False))
    print(f"selected {NUM_EXAMPLES} of {n_val} validation examples "
          f"(selection_seed={SELECTION_SEED}):\n{val_indices}")

    sel_max = int(data["n_sources"][val_indices].max())
    n_src_pad = sel_max if MAX_N_SOURCES is None else int(MAX_N_SOURCES)
    print(f"padded source-cloud length n_src_pad={n_src_pad} "
          f"(selected-example max={sel_max})")
    if n_src_pad < sel_max:
        raise ValueError(
            f"MAX_N_SOURCES={n_src_pad} < largest selected example ({sel_max}); "
            "raise MAX_N_SOURCES or set it to None"
        )

    param_path = os.path.join(MODEL_DIR, f"cond_diffusion_params{SUFFIX}.pkl")
    sched_path = os.path.join(MODEL_DIR, f"cond_diffusion_schedule{SUFFIX}.pkl")
    with open(param_path, "rb") as f:
        params = pickle.load(f)
    with open(sched_path, "rb") as f:
        diffusion_cfg = pickle.load(f)

    betas = jnp.asarray(diffusion_cfg["betas"], dtype=jnp.float32)
    alphas = jnp.asarray(diffusion_cfg["alphas"], dtype=jnp.float32)
    alpha_bars = jnp.asarray(diffusion_cfg["alpha_bars"], dtype=jnp.float32)
    print(f"loaded params from {param_path}; T={betas.shape[0]} diffusion steps")

    cfg = DiffusionModelConfig(
        base_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        out_channels=1,
        galaxy_token_dim=128,
        num_attention_heads=4,
        coord_image_size=16,
        src_embed_dim=SRC_EMBED_DIM,
        src_out_ch=SRC_OUT_CH,
        shape_grid_size=grid_size,
    )
    model = ConditionalUNet3D(cfg=cfg)

    run_sampling(
        model, params, data, betas, alphas, alpha_bars,
        val_indices=val_indices,
        n_draws=N_POSTERIOR_SAMPLES,
        batch_size=SAMPLING_BATCH_SIZE,
        n_src_pad=n_src_pad,
        grid_size=grid_size,
        out_dir=OUT_DIR,
        base_seed=BASE_SAMPLE_SEED,
    )


if __name__ == "__main__":
    main()
