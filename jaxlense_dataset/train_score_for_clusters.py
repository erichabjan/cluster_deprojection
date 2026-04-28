#!/usr/bin/env python3
"""
Stage 2a: train the UResNet18 score network on (mean_beta * kappa_inf) maps from BAHAMAS clusters.

Adapted from jax_lensing/scripts/train_score.py:
  - reads kappa maps directly from per-sample npz files (no tensorflow_datasets dependency)
  - works at our resolution (default 128) and at our cluster prior
  - skips tensorboard (just stdout logging) so we never touch core tensorflow at runtime

Run: jax_lense env, GPU.
"""
import os
import sys
import time
import pickle

import numpy as onp
sys.path.insert(0, "/home/habjan.e/TNG/Codes/jax-lensing")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CFG_DATA, CFG_GRID, CFG_LENS, CFG_COSMO, CFG_SCORE

import jax
import jax.numpy as jnp
import haiku as hk

from jax_lensing.models import UResNet18
from jax_lensing.models.normalization import SNParamsTree
from jax_lensing.spectral import make_power_map


# Manual Adam (avoids optax's stale haiku FlatMap pytree registration which collides
# with modern haiku returning regular dicts in 0.0.11).
def adam_init(params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return (m, v, jnp.zeros((), jnp.int32))


def adam_update(grads, state, lr=1e-4, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1
    m = jax.tree_util.tree_map(lambda mi, gi: b1 * mi + (1.0 - b1) * gi, m, grads)
    v = jax.tree_util.tree_map(lambda vi, gi: b2 * vi + (1.0 - b2) * gi * gi, v, grads)
    bc1 = 1.0 - b1 ** t
    bc2 = 1.0 - b2 ** t
    updates = jax.tree_util.tree_map(
        lambda mi, vi: -lr * (mi / bc1) / (jnp.sqrt(vi / bc2) + eps), m, v)
    return updates, (m, v, t)


def apply_updates(params, updates):
    return jax.tree_util.tree_map(lambda p, u: p + u, params, updates)


def kappa_iterator(sample_ids, interim_dir, mean_beta, batch_size, map_size,
                   flip_aug=True, rng_seed=0):
    """Yield batches of clean kappa maps shape (B, M, M, 1) as float32 numpy arrays."""
    rng = onp.random.default_rng(rng_seed)
    paths = [os.path.join(interim_dir, f"sample_{int(s):06d}.npz") for s in sample_ids]
    paths = [p for p in paths if os.path.exists(p)]
    assert len(paths) > 0, "no kappa training files found; run stage 1 first"
    while True:
        rng.shuffle(paths)
        for i in range(0, len(paths) - batch_size + 1, batch_size):
            batch = []
            for p in paths[i:i + batch_size]:
                d = onp.load(p)
                k = (mean_beta * d["kappa_inf"]).astype(onp.float32)
                if k.shape[0] != map_size:
                    raise RuntimeError(
                        f"map_size mismatch: {k.shape} vs requested {map_size}")
                if flip_aug:
                    if rng.random() < 0.5:
                        k = k[::-1, :]
                    if rng.random() < 0.5:
                        k = k[:, ::-1]
                batch.append(k[..., None])
            yield onp.stack(batch, axis=0)


def forward_fn(x, s, is_training=False):
    return UResNet18(n_output_channels=1)(x, s, is_training=is_training)


def main():
    map_size = CFG_GRID.lens_recon_resolution
    out_dir = os.path.join(CFG_DATA.output_root, CFG_DATA.weights_dirname)
    interim_dir = os.path.join(CFG_DATA.output_root, CFG_DATA.intermediate_dirname)
    os.makedirs(out_dir, exist_ok=True)

    manifest = onp.load(os.path.join(CFG_DATA.output_root, "manifest.npz"))
    train_ids = manifest["train_ids"]
    n_use = int(len(train_ids) * CFG_SCORE.train_split_fraction)
    train_ids = train_ids[:n_use]
    mean_beta = float(onp.load(os.path.join(out_dir, "mean_beta.npy")))
    pixel_size_rad = float(onp.load(os.path.join(out_dir, "pixel_size_rad.npy")))
    print(f"map_size={map_size}, mean_beta={mean_beta:.6f}, pixel_rad={pixel_size_rad:.4e}, "
          f"using {len(train_ids)} train samples", flush=True)

    ps_data = onp.load(os.path.join(out_dir, "cluster_kappa_PS_theory.npy")).astype("float32")
    ell = jnp.array(ps_data[0, :])
    ps_halofit = jnp.array(ps_data[1, :] / pixel_size_rad ** 2)
    kell = ell / 2.0 / jnp.pi * 360.0 * pixel_size_rad / map_size
    power_map = jnp.array(make_power_map(ps_halofit, map_size, kps=kell))

    def log_gaussian_prior(map_data, sigma, ps_map):
        data_ft = jnp.fft.fft2(map_data) / float(map_size)
        return -0.5 * jnp.sum(jnp.real(data_ft * jnp.conj(data_ft)) / (ps_map + sigma[0] ** 2))
    gaussian_prior_score = jax.vmap(jax.grad(log_gaussian_prior), in_axes=[0, 0, None])

    model = hk.without_apply_rng(hk.transform_with_state(forward_fn))
    sn_fn = hk.transform_with_state(
        lambda x: SNParamsTree(ignore_regex='[^?!.]*b$|[^?!.]*offset$',
                               val=CFG_SCORE.spectral_norm)(x))

    rng_seq = hk.PRNGSequence(42)
    params, state = model.init(next(rng_seq),
                               jnp.zeros((1, map_size, map_size, 2)),
                               jnp.zeros((1, 1, 1, 1)), is_training=True)
    opt_state = adam_init(params)
    _, sn_state = sn_fn.init(next(rng_seq), params)

    def score_fn(params, state, batch, is_training=True):
        gs = gaussian_prior_score(batch['y'][..., 0], batch['s'][..., 0], power_map)
        gs = jnp.expand_dims(gs, axis=-1)
        net_input = jnp.concatenate([batch['y'], jnp.abs(batch['s']) ** 2 * gs], axis=-1)
        res, state = model.apply(params, state, net_input, batch['s'], is_training=is_training)
        return res, state, gs

    def loss_fn(params, state, batch):
        res, state, gs = score_fn(params, state, batch)
        loss = jnp.mean((batch['u'] + batch['s'] * (res + gs)) ** 2)
        return loss, state

    params_treedef = jax.tree_util.tree_structure(params)

    @jax.jit
    def update(params, state, sn_state, opt_state, batch):
        (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, batch)
        # haiku 0.0.11 + chex 0.1.90: value_and_grad returns FlatMap-shaped grads even
        # when params is a regular dict; reflatten to params' structure so manual Adam's
        # tree_map across (m, grads) sees a single consistent pytree type.
        grads = jax.tree_util.tree_unflatten(params_treedef, jax.tree_util.tree_leaves(grads))
        updates, opt_state = adam_update(grads, opt_state, lr=CFG_SCORE.learning_rate)
        params = apply_updates(params, updates)
        params, sn_state = sn_fn.apply(None, sn_state, None, params)
        return loss, params, state, sn_state, opt_state

    iterator = kappa_iterator(train_ids, interim_dir, mean_beta,
                              CFG_SCORE.batch_size, map_size)
    rng_np = onp.random.default_rng(123)

    print("training begins.", flush=True)
    t0 = time.time()
    for step in range(CFG_SCORE.training_steps):
        x_np = next(iterator)
        u_np = rng_np.standard_normal(x_np.shape).astype(onp.float32)
        s_np = (CFG_SCORE.noise_dist_std
                * rng_np.standard_normal((x_np.shape[0], 1, 1, 1))).astype(onp.float32)
        y_np = x_np + s_np * u_np
        batch = {'x': jnp.asarray(x_np), 'y': jnp.asarray(y_np),
                 'u': jnp.asarray(u_np), 's': jnp.asarray(s_np)}
        loss, params, state, sn_state, opt_state = update(
            params, state, sn_state, opt_state, batch)

        if step % CFG_SCORE.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d} | loss {float(loss):.6f} | "
                  f"elapsed {elapsed/60:.2f} min", flush=True)

        if step > 0 and step % CFG_SCORE.checkpoint_every == 0:
            ckpt_path = os.path.join(out_dir, f"score_model-{step}.pckl")
            with open(ckpt_path, 'wb') as fh:
                pickle.dump([params, state, sn_state], fh)

    final_path = os.path.join(out_dir, "score_model-final.pckl")
    with open(final_path, 'wb') as fh:
        pickle.dump([params, state, sn_state], fh)
    print(f"saved final weights: {final_path}")
    print("stage 2a done.")


if __name__ == "__main__":
    main()
