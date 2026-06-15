#!/usr/bin/env python3
"""
Apply the trained jax-lensing score-based posterior to the Abell 3411 SuperBIT
shape catalog and produce a posterior convergence (kappa_E) map.

Inputs
------
SuperBIT shape catalog (FITS):
    ra, dec                : sky coordinates (deg, J2000)
    g1_cal_5x5, g2_cal_5x5 : metacal-calibrated reduced-shear components
    w_inv_5x5              : per-galaxy inverse-variance weight (confirmed
                             by the data: median(1/sqrt(w_inv_5x5)) ~ 0.17,
                             consistent with intrinsic shape noise sigma_e ~ 0.2)
Catalog header gives the cluster centre via RA_CNTR, DEC_CNTR.

Trained prior artifacts (written by stage 2 prep + stage 2a):
    {weights_dir}/score_model-final.pckl
    {weights_dir}/cluster_kappa_PS_theory.npy
    {weights_dir}/pixel_size_rad.npy
    {weights_dir}/mean_beta.npy
These were computed at z_l=0.2 (the training fiducial); we keep them as-is.

Strategy
--------
* Tangent-project the source catalog onto a 128x128 grid centred on
  (RA_CNTR, DEC_CNTR) from the catalog header.
* The grid spans the *same physical pixel size* as training (78 kpc), but
  computed at A3411's actual z_l=0.169. This keeps the prior's per-pixel
  power spectrum self-consistent (the score net sees only pixel indices,
  not angles), while preserving the cluster's physical morphology scale.
* Inverse-variance-bin per-galaxy shears with w_inv_5x5 to produce per-pixel
  weighted-mean gamma1_obs, gamma2_obs, and n_eff_pix = (sum w)^2 / sum w^2.
* Per-pixel shear noise: sigma_e_eff / sqrt(n_eff_pix), with sigma_e_eff
  estimated from the catalog as sqrt(<1/w_i>) so the noise model is
  consistent with the actual metacal calibration rather than the training
  fiducial sigma_e=0.2.
* The geometric factor <beta> is recomputed at z_l=0.169 over the SuperBIT
  selected_redshift template (the n(z) used to build the mocks) and saved
  as metadata. The score sampler operates on raw shear / kappa units, so
  this is purely informational for downstream Sigma / mass conversion.

Outputs
-------
NPZ at OUTPUT_PATH with:
  kappa_E_mean, kappa_E_std         per-pixel posterior moments
  kappa_E_samples                   full (n_chains, 128, 128) sample stack so
                                    downstream code can compute exact smoothed
                                    SNR (smooth each chain, then take std)
                                    instead of an independence-assumption bracket
  gamma1_obs, gamma2_obs            inverse-variance-binned input shears
  n_eff_pix, mask                   per-pixel effective N + 0/1 data footprint
  metadata                          z_l, ra0/dec0, FoV, mean_beta_*, sigma_e_eff,
                                    Sigma_crit_inf, pixel-scale arrays

Run: jax_lense env, GPU. Companion: abell3411_jaxlensing.slurm.
"""
import os
import sys
import time
import argparse
from dataclasses import replace

import numpy as onp
from astropy.io import fits
from scipy import integrate

sys.path.insert(0, "/home/habjan.e/TNG/Codes/jax-lensing")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CFG_DATA, CFG_GRID, CFG_LENS, CFG_COSMO, CFG_DLP
import lensing_utils as lu
from run_dlposterior import build_score_pipeline

import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
from jax_lensing.samplers.score_samplers import ScoreHamiltonianMonteCarlo
from jax_lensing.samplers.tempered_sampling import TemperedMC
from jax_lensing.inversion import ks93, ks93inv


CATALOG_PATH_DEFAULT = ("/home/habjan.e/TNG/Data/superbit_redshifts/"
                        "Abell_3411/Abell3411_shapes_cal_selected.fits")
OUTPUT_PATH_DEFAULT  = os.path.join(CFG_DATA.output_root, "posterior_abell3411.npz")
Z_L_A3411            = 0.169       # NED redshift


# ---------------------------------------------------------------------------
# Catalog ingest + projection
# ---------------------------------------------------------------------------

def load_catalog(path):
    with fits.open(path) as hdul:
        t = hdul[1].data
        hdr = hdul[1].header
        ra  = onp.asarray(t["ra"],         dtype=onp.float64)
        dec = onp.asarray(t["dec"],        dtype=onp.float64)
        g1  = onp.asarray(t["g1_cal_5x5"], dtype=onp.float64)
        g2  = onp.asarray(t["g2_cal_5x5"], dtype=onp.float64)
        w   = onp.asarray(t["w_inv_5x5"],  dtype=onp.float64)
        ra0  = float(hdr["RA_CNTR"])
        dec0 = float(hdr["DEC_CNTR"])
    return ra, dec, g1, g2, w, ra0, dec0


def tangent_project(ra, dec, ra0, dec0):
    """Flat-sky gnomonic projection. Returns (x, y) in arcmin with east-positive
    x and north-positive y, valid for sub-degree fields."""
    x_arcmin = (ra - ra0) * onp.cos(onp.deg2rad(dec0)) * 60.0
    y_arcmin = (dec - dec0) * 60.0
    return x_arcmin, y_arcmin


# ---------------------------------------------------------------------------
# Per-pixel inverse-variance binning
# ---------------------------------------------------------------------------

def bin_shears_to_grid(x_arcmin, y_arcmin, g1, g2, w, half_fov_arcmin, N):
    """Inverse-variance-bin per-galaxy shears onto an NxN grid.

    Convention matches lensing_utils.bin2d_sum: np.histogram2d(y, x, ...) so
    arr[iy, ix] indexes (y, x), which is also the convention build_one_sample
    uses for gamma1_obs / gamma2_obs in the training mocks.
    """
    edges = onp.linspace(-half_fov_arcmin, half_fov_arcmin, N + 1)
    bins = (edges, edges)
    sum_w,   _, _ = onp.histogram2d(y_arcmin, x_arcmin, bins=bins, weights=w)
    sum_wg1, _, _ = onp.histogram2d(y_arcmin, x_arcmin, bins=bins, weights=w * g1)
    sum_wg2, _, _ = onp.histogram2d(y_arcmin, x_arcmin, bins=bins, weights=w * g2)
    sum_w2,  _, _ = onp.histogram2d(y_arcmin, x_arcmin, bins=bins, weights=w * w)

    mask = sum_w > 0
    denom_w  = onp.where(mask, sum_w,  1.0)
    denom_w2 = onp.where(mask, sum_w2, 1.0)
    g1_pix = onp.where(mask, sum_wg1 / denom_w,         0.0).astype(onp.float32)
    g2_pix = onp.where(mask, sum_wg2 / denom_w,         0.0).astype(onp.float32)
    n_eff  = onp.where(mask, sum_w * sum_w / denom_w2,  0.0).astype(onp.float32)
    return g1_pix, g2_pix, n_eff, mask.astype(onp.float32)


# ---------------------------------------------------------------------------
# Posterior sampler on in-memory arrays
# ---------------------------------------------------------------------------

def run_posterior_on_arrays(g1_obs, g2_obs, n_eff_pix, mask, sigma_e_eff,
                             score_prior_fn, map_size, dlp):
    """Mirror of run_dlposterior.run_one_sample but with arrays in memory and a
    configurable per-component shape-noise sigma_e_eff (so the noise model can
    track the actual metacal calibration rather than the training fiducial)."""
    g1_obs = g1_obs.astype(onp.float32)
    g2_obs = g2_obs.astype(onp.float32)
    n_pix  = n_eff_pix.astype(onp.float32)
    mask_np = mask.astype(onp.float32)
    std_np = onp.where(
        mask_np > 0,
        sigma_e_eff / onp.sqrt(onp.maximum(n_pix, 1.0)),
        1.0,
    ).astype(onp.float32)

    masked_shear = jnp.stack(
        [jnp.array(g1_obs * mask_np), jnp.array(g2_obs * mask_np)], axis=-1)
    sigma_gamma = jnp.stack([jnp.array(std_np), jnp.array(std_np)], axis=-1)
    sigma_mask  = jnp.array((1.0 - mask_np) * 1e10)[..., None]

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

    # Probability-flow ODE drift for post-HMC denoising (Section 4 of
    # Remy et al. 2022; sample_hmc.py:286-308 in jax-lensing). Each TempMC
    # sample is at residual noise sigma ~ min_temperature; integrating
    # dx/dt = -1/2 * score(x, sqrt(t)) from t = sigma^2 down to t = 1e-5
    # transports it back to the (clean) posterior.
    @jax.jit
    def ode_drift_jit(t, x_flat):
        x = x_flat.reshape((dlp.batch_size, map_size * map_size))
        sigma = jnp.ones((dlp.batch_size,)) * jnp.sqrt(t)
        return (-0.5 * total_score_fn(x, sigma)).reshape(-1)

    def ode_drift(t, x_flat):
        # scipy.integrate.solve_ivp passes y as float64; pin dtypes to keep
        # the jit cache stable and avoid silent precision loss in the net.
        out = ode_drift_jit(jnp.float32(t),
                            jnp.asarray(x_flat, dtype=jnp.float32))
        return onp.asarray(out, dtype=onp.float64)

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

        # Trace the post-tempering temperatures so we can start the ODE at the
        # actual final noise level (defensive — with the configured schedule
        # the chains do reach min_temperature, but we don't want to assume).
        samples, trace_temps = tfp.mcmc.sample_chain(
            num_results=1,
            current_state=x0,
            kernel=tmc,
            num_burnin_steps=0,
            num_steps_between_results=dlp.num_steps_between_results,
            trace_fn=lambda _, pkr: pkr.post_tempering_inverse_temperatures,
            seed=jax.random.PRNGKey(seed_i + 1000))

        last_temp = float(jnp.mean(trace_temps[-1]))
        noise = max(last_temp, float(dlp.min_temperature))
        t_start = 0.99 * noise ** 2
        t_end = 1e-5
        hmc_sample_flat = onp.asarray(samples[-1]).reshape(-1)
        # Omit t_eval: solve_ivp lands its final step exactly at t_span[1],
        # so sol.y[:, -1] is the denoised sample at t_end. (Passing an explicit
        # t_eval with log-spaced endpoints fails scipy's bounds check because
        # the log10/10**x round-trip nudges the endpoints just outside t_span.)
        sol = integrate.solve_ivp(
            fun=ode_drift,
            t_span=(t_start, t_end),
            y0=hmc_sample_flat,
            method="RK45",
        )
        if not sol.success:
            raise RuntimeError(f"ODE denoising failed: {sol.message}")
        denoised = sol.y[:, -1].reshape(dlp.batch_size, map_size, map_size)
        samples_all.append(denoised.astype(onp.float32))
        print(f"  seed {seed_i + 1}/{dlp.n_independent_seeds}: "
              f"last_temp={last_temp:.4f}, ODE nfev={sol.nfev}, "
              f"{dlp.batch_size} chains done", flush=True)

    samples_all = onp.concatenate(samples_all, axis=0).astype(onp.float32)
    return (samples_all.mean(axis=0).astype(onp.float32),
            samples_all.std(axis=0).astype(onp.float32),
            samples_all,
            int(samples_all.shape[0]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog",    default=CATALOG_PATH_DEFAULT)
    ap.add_argument("--output",     default=OUTPUT_PATH_DEFAULT)
    ap.add_argument("--z_l",        type=float, default=Z_L_A3411)
    ap.add_argument("--batch_size", type=int,   default=CFG_DLP.batch_size)
    # 8x the original n_independent_seeds default to reduce MC noise on the
    # per-pixel posterior mean. With batch_size=100 this yields 3200 total
    # chains; the outer-loop wall-time scales linearly with n_seeds.
    ap.add_argument("--n_seeds",    type=int,   default=32)
    args = ap.parse_args()

    dlp = replace(CFG_DLP,
                  batch_size=args.batch_size,
                  n_independent_seeds=args.n_seeds)

    print(f"abell3411_jaxlensing\n  catalog = {args.catalog}\n  output  = {args.output}", flush=True)

    # 1. Catalog and projection.
    ra, dec, g1, g2, w, ra0, dec0 = load_catalog(args.catalog)
    x_arcmin, y_arcmin = tangent_project(ra, dec, ra0, dec0)
    print(f"  N_src = {len(ra)}, centre = ({ra0:.4f}, {dec0:.4f}) deg", flush=True)
    print(f"  catalog extent: x in [{x_arcmin.min():+.2f}, {x_arcmin.max():+.2f}] arcmin, "
          f"y in [{y_arcmin.min():+.2f}, {y_arcmin.max():+.2f}] arcmin", flush=True)

    # 2. Grid: preserve training's physical pixel size (78 kpc) at A3411's z_l.
    map_size = CFG_GRID.lens_recon_resolution            # 128
    fov_mpc  = CFG_GRID.fov_mpc                          # 5 Mpc half-extent
    z_l = args.z_l
    D_l = lu.angular_diameter_distance_mpc(z_l, H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0)
    arcmin_per_mpc = (180.0 * 60.0 / onp.pi) / D_l
    half_fov_arcmin         = fov_mpc * arcmin_per_mpc
    pixel_size_arcmin_data  = (2.0 * fov_mpc / map_size) * arcmin_per_mpc
    print(f"  z_l = {z_l}, D_l = {D_l:.1f} Mpc, "
          f"half_FoV = {half_fov_arcmin:.2f} arcmin, "
          f"pixel = {pixel_size_arcmin_data:.4f} arcmin", flush=True)

    # 3. Bin shears.
    g1_obs, g2_obs, n_eff_pix, mask = bin_shears_to_grid(
        x_arcmin, y_arcmin, g1, g2, w, half_fov_arcmin, map_size)
    n_data_pix = int(mask.sum())
    pop = n_eff_pix[mask > 0]
    print(f"  n_data_pix = {n_data_pix}/{map_size * map_size} "
          f"({100 * n_data_pix / (map_size * map_size):.1f}% of grid)", flush=True)
    print(f"  n_eff_pix among populated: median={onp.median(pop):.2f}, "
          f"min={pop.min():.2f}, max={pop.max():.2f}", flush=True)

    # 4. Effective per-source shape noise from the catalog weights.
    sigma_e_eff = float(onp.sqrt(onp.mean(1.0 / w)))
    print(f"  sigma_e_eff = sqrt(<1/w>) = {sigma_e_eff:.4f}  "
          f"(training fiducial = {CFG_LENS.sigma_e_per_component})", flush=True)

    # 5. Geometric factor: <beta> at A3411 z_l over the SuperBIT n(z) template.
    z_template = onp.asarray(
        onp.load(CFG_LENS.empirical_redshift_npz_path)[CFG_LENS.empirical_redshift_key],
        dtype=onp.float64)
    mean_beta_a3411 = lu.mean_beta_over_empirical(
        z_l, z_template, H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0)
    weights_dir = os.path.join(CFG_DATA.output_root, CFG_DATA.weights_dirname)
    mean_beta_training      = float(onp.load(os.path.join(weights_dir, "mean_beta.npy")))
    pixel_size_rad_training = float(onp.load(os.path.join(weights_dir, "pixel_size_rad.npy")))
    sigma_crit_inf_a3411    = lu.sigma_crit_inf_msun_per_mpc2(
        z_l, H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0)
    print(f"  mean_beta_a3411  (z_l={z_l})    = {mean_beta_a3411:.4f}", flush=True)
    print(f"  mean_beta_train  (z_l=0.2 fid.) = {mean_beta_training:.4f}", flush=True)
    print(f"  Sigma_crit_inf(z_l={z_l})       = {sigma_crit_inf_a3411:.3e} Msun/Mpc^2",
          flush=True)

    # 6. Score-prior pipeline. Use the *training* pixel_size_rad so the
    #    Gaussian-prior per-pixel P(k) matches what the score net saw.
    weights_path = os.path.join(weights_dir, "score_model-final.pckl")
    ps_path      = os.path.join(weights_dir, "cluster_kappa_PS_theory.npy")
    score_prior_fn = build_score_pipeline(
        map_size, weights_path, ps_path, pixel_size_rad_training)

    # 7. Posterior sampling.
    print(f"posterior sampling: batch={dlp.batch_size}, "
          f"seeds={dlp.n_independent_seeds}, "
          f"total chains = {dlp.batch_size * dlp.n_independent_seeds}", flush=True)
    t0 = time.time()
    kappa_E_mean, kappa_E_std, kappa_E_samples, n_samples = run_posterior_on_arrays(
        g1_obs, g2_obs, n_eff_pix, mask, sigma_e_eff,
        score_prior_fn, map_size, dlp)
    print(f"  done {n_samples} posterior samples in {(time.time() - t0)/60:.2f} min", flush=True)
    print(f"  kappa_E_samples shape={kappa_E_samples.shape}, "
          f"size={kappa_E_samples.nbytes / 1e6:.1f} MB (uncompressed)", flush=True)

    # 8. Save.
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    onp.savez_compressed(
        args.output,
        kappa_E_mean=kappa_E_mean,
        kappa_E_std=kappa_E_std,
        kappa_E_samples=kappa_E_samples,
        gamma1_obs=g1_obs.astype(onp.float32),
        gamma2_obs=g2_obs.astype(onp.float32),
        n_eff_pix=n_eff_pix.astype(onp.float32),
        mask=mask.astype(onp.float32),
        mean_beta_training=onp.float32(mean_beta_training),
        mean_beta_a3411=onp.float32(mean_beta_a3411),
        sigma_e_eff_per_component=onp.float32(sigma_e_eff),
        sigma_crit_inf_a3411=onp.float64(sigma_crit_inf_a3411),
        z_l=onp.float32(z_l),
        ra0=onp.float64(ra0),
        dec0=onp.float64(dec0),
        fov_mpc=onp.float32(fov_mpc),
        half_fov_arcmin=onp.float32(half_fov_arcmin),
        pixel_size_arcmin_data=onp.float32(pixel_size_arcmin_data),
        pixel_size_rad_training=onp.float32(pixel_size_rad_training),
        n_posterior_samples=onp.int32(n_samples),
    )
    print(f"wrote {args.output}")
    print("abell3411_jaxlensing done.")


if __name__ == "__main__":
    main()
