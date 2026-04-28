#!/usr/bin/env python3
"""
Stage 2b: posterior-sample kappa_E from binned shear (gamma1_obs, gamma2_obs) using DLPosterior.

Adapted from jax_lensing/scripts/sample_hmc.py:
  - reads per-sample inputs from stage-1 npz files (no fits I/O)
  - operates at our resolution (default 128)
  - reads our cluster-trained score-net weights and our cluster Gaussian prior P(k)
  - writes posterior mean and std to {output_root}/{posterior_dirname}/posterior_{NNNNNN}.npz

Supports a --start / --stop range for parallelization across multiple SLURM jobs (or array jobs).
Run: jax_lense env, GPU.
"""
import os
import sys
import time
import pickle
import argparse
from functools import partial

import numpy as onp
sys.path.insert(0, "/home/habjan.e/TNG/Codes/jax-lensing")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CFG_DATA, CFG_GRID, CFG_LENS, CFG_DLP

import jax
import jax.numpy as jnp
import haiku as hk

# tensorflow_probability via the jax substrate (avoids importing core tensorflow)
from tensorflow_probability.substrates import jax as tfp

from jax_lensing.samplers.score_samplers import ScoreHamiltonianMonteCarlo
from jax_lensing.samplers.tempered_sampling import TemperedMC
from jax_lensing.models.convdae import UResNet18
from jax_lensing.spectral import make_power_map
from jax_lensing.inversion import ks93inv, ks93


def forward_fn(x, s, is_training=False):
    return UResNet18(n_output_channels=1)(x, s, is_training=is_training)


def build_score_pipeline(map_size, weights_path, ps_path, pixel_size_rad):
    """Compose the score function (Gaussian prior + denoiser); returns score_prior(x, sigma)."""
    ps_data = onp.load(ps_path).astype('float32')
    ell = jnp.array(ps_data[0, :])
    ps_halofit = jnp.array(ps_data[1, :] / pixel_size_rad ** 2)
    kell = ell / 2.0 / jnp.pi * 360.0 * pixel_size_rad / map_size
    power_map = jnp.array(make_power_map(ps_halofit, map_size, kps=kell))

    def log_gaussian_prior(m, s, ps_map):
        data_ft = jnp.fft.fft2(m) / float(map_size)
        return -0.5 * jnp.sum(jnp.real(data_ft * jnp.conj(data_ft)) / (ps_map + s ** 2))
    gaussian_prior_score = jax.vmap(jax.grad(log_gaussian_prior), in_axes=[0, 0, None])

    model = hk.without_apply_rng(hk.transform_with_state(forward_fn))
    rng_seq = hk.PRNGSequence(42)
    params, state = model.init(next(rng_seq),
                               jnp.zeros((1, map_size, map_size, 2)),
                               jnp.zeros((1, 1, 1, 1)), is_training=True)
    with open(weights_path, 'rb') as fh:
        params, state, _ = pickle.load(fh)
    score_apply = partial(model.apply, params, state, is_training=False)

    def score_prior(x, sigma):
        ke = x.reshape((-1, map_size, map_size))
        gs = gaussian_prior_score(ke, sigma.reshape((-1, 1, 1)), power_map)
        gs = jnp.expand_dims(gs, axis=-1)
        net_input = jnp.concatenate(
            [ke.reshape((-1, map_size, map_size, 1)),
             jnp.abs(sigma.reshape((-1, 1, 1, 1))) ** 2 * gs], axis=-1)
        res, _ = score_apply(net_input, sigma.reshape((-1, 1, 1, 1)))
        return (res[..., 0] + gs[..., 0]).reshape(-1, map_size * map_size)

    return score_prior


def run_one_sample(in_path, score_prior_fn, map_size, dlp):
    d = onp.load(in_path)
    g1_obs = d["gamma1_obs"].astype(onp.float32)
    g2_obs = d["gamma2_obs"].astype(onp.float32)
    n_pix = d["n_gal_pix"].astype(onp.float32)
    sigma_e = CFG_LENS.sigma_e_per_component

    mask_np = (n_pix > 0).astype(onp.float32)
    std_np = onp.where(n_pix > 0, sigma_e / onp.sqrt(onp.maximum(n_pix, 1.0)), 1.0).astype(onp.float32)

    masked_shear = jnp.stack([jnp.array(g1_obs * mask_np), jnp.array(g2_obs * mask_np)], axis=-1)
    sigma_gamma = jnp.stack([jnp.array(std_np), jnp.array(std_np)], axis=-1)
    sigma_mask = jnp.array((1.0 - mask_np) * 1e10)[..., None]

    def log_likelihood(x, sigma_temp, meas_shear, sigma_mask_arr):
        ke = x.reshape((map_size, map_size))
        kb = jnp.zeros_like(ke)
        gm = jnp.stack(ks93inv(ke, kb), axis=-1)
        return -jnp.sum((gm - meas_shear) ** 2
                        / ((sigma_gamma ** 2) + sigma_temp ** 2 + sigma_mask_arr)) / 2.0
    likelihood_score = jax.vmap(jax.grad(log_likelihood), in_axes=[0, 0, None, None])

    def total_score_fn(x, sigma):
        sl = likelihood_score(x, sigma, masked_shear, sigma_mask).reshape(-1, map_size * map_size)
        sp = score_prior_fn(x, sigma)
        return (sl + sp).reshape(-1, map_size * map_size)

    init_image_2d, _ = ks93(masked_shear[..., 0], masked_shear[..., 1])
    init_image = jnp.broadcast_to(init_image_2d, (dlp.batch_size, map_size, map_size))

    def make_kernel_fn(target_log_prob_fn, target_score_fn, sigma):
        return ScoreHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            target_score_fn=target_score_fn,
            step_size=dlp.initial_step_size * (jnp.max(sigma) / dlp.initial_temperature) ** 0.5,
            num_leapfrog_steps=3,
            num_delta_logp_steps=4)

    samples_all = []
    for seed_i in range(dlp.n_independent_seeds):
        x0 = init_image + dlp.initial_temperature * jax.random.normal(
            jax.random.PRNGKey((int(time.time() * 1000) + seed_i) % (2 ** 31)),
            (dlp.batch_size, map_size, map_size))
        x0 = x0.reshape(dlp.batch_size, -1)

        tmc = TemperedMC(
            target_score_fn=total_score_fn,
            inverse_temperatures=dlp.initial_temperature * jnp.ones([dlp.batch_size]),
            make_kernel_fn=make_kernel_fn,
            gamma=dlp.cooling_gamma,
            min_temp=dlp.min_temperature,
            min_steps_per_temp=dlp.min_steps_per_temp,
            num_delta_logp_steps=4)

        samples = tfp.mcmc.sample_chain(
            num_results=1,
            current_state=x0,
            kernel=tmc,
            num_burnin_steps=0,
            num_steps_between_results=dlp.num_steps_between_results,
            trace_fn=None,
            seed=jax.random.PRNGKey(seed_i + 1000))
        samples_all.append(onp.asarray(samples[-1].reshape(dlp.batch_size, map_size, map_size)))

    samples_all = onp.concatenate(samples_all, axis=0)
    return (samples_all.mean(axis=0).astype(onp.float32),
            samples_all.std(axis=0).astype(onp.float32),
            int(samples_all.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=-1)
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="re-run even if output already exists")
    args = ap.parse_args()

    map_size = CFG_GRID.lens_recon_resolution
    weights_dir = os.path.join(CFG_DATA.output_root, CFG_DATA.weights_dirname)
    interim_dir = os.path.join(CFG_DATA.output_root, CFG_DATA.intermediate_dirname)
    posterior_dir = os.path.join(CFG_DATA.output_root, CFG_DATA.posterior_dirname)
    os.makedirs(posterior_dir, exist_ok=True)

    weights_path = os.path.join(weights_dir, "score_model-final.pckl")
    ps_path = os.path.join(weights_dir, "cluster_kappa_PS_theory.npy")
    mean_beta = float(onp.load(os.path.join(weights_dir, "mean_beta.npy")))
    pixel_size_rad = float(onp.load(os.path.join(weights_dir, "pixel_size_rad.npy")))

    score_prior_fn = build_score_pipeline(map_size, weights_path, ps_path, pixel_size_rad)

    manifest = onp.load(os.path.join(CFG_DATA.output_root, "manifest.npz"))
    all_ids = onp.concatenate([manifest["train_ids"], manifest["test_ids"]])
    all_ids.sort()
    stop = len(all_ids) if args.stop < 0 else args.stop
    targets = all_ids[args.start:stop]
    print(f"posterior sampling {len(targets)} samples (start={args.start}, stop={stop}); "
          f"map_size={map_size}, batch={CFG_DLP.batch_size}, "
          f"seeds={CFG_DLP.n_independent_seeds}, mean_beta={mean_beta:.4f}", flush=True)

    skip_existing = not args.no_skip_existing
    t0 = time.time()
    n_done = 0
    for k, sid in enumerate(targets):
        out_path = os.path.join(posterior_dir, f"posterior_{int(sid):06d}.npz")
        if skip_existing and os.path.exists(out_path):
            continue
        in_path = os.path.join(interim_dir, f"sample_{int(sid):06d}.npz")
        if not os.path.exists(in_path):
            print(f"  missing input {in_path}, skipping")
            continue

        kE_mean, kE_std, n_samples = run_one_sample(in_path, score_prior_fn, map_size, CFG_DLP)
        onp.savez_compressed(
            out_path,
            kappa_E_mean=kE_mean,
            kappa_E_std=kE_std,
            mean_beta=onp.float32(mean_beta),
            n_posterior_samples=onp.int32(n_samples),
        )
        n_done += 1
        if n_done % 5 == 0 or n_done == len(targets):
            elapsed = time.time() - t0
            avg = elapsed / max(n_done, 1)
            print(f"  done {n_done}/{len(targets)} (avg {avg:.1f} s/sample, "
                  f"elapsed {elapsed/60:.1f} min)", flush=True)

    print("stage 2b done.")


if __name__ == "__main__":
    main()
