#!/usr/bin/env python3
"""
Stage 2 prep: compute the average P(k) of (mean_beta * kappa_inf) over the training set,
plus the constant <beta>. Stores both alongside the score-net weights so that
train_score_for_clusters.py and run_dlposterior.py can load them.

The P(k) file format mirrors what jax_lensing.spectral.measure_power_spectrum produces
and what train_score.py / sample_hmc.py consume:
    np.array([ell, P(ell)], dtype=float64), shape (2, N_ell)
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CFG_DATA, CFG_GRID, CFG_LENS, CFG_COSMO
import lensing_utils as lu


def _radial_profile(arr2d):
    cy, cx = arr2d.shape[0] / 2.0, arr2d.shape[1] / 2.0
    y, x = np.indices(arr2d.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), arr2d.ravel())
    nr = np.bincount(r.ravel())
    nr = np.where(nr == 0, 1, nr)
    return tbin / nr


def measure_ps(kappa_map, pixel_size_rad):
    """Match jax_lensing.spectral.measure_power_spectrum: returns (ell, P(ell))."""
    N = kappa_map.shape[0]
    ft = np.fft.fftshift(np.fft.fft2(kappa_map)) / N
    nyq = N // 2
    p1d = _radial_profile(np.real(ft * np.conj(ft)))[:nyq] * (pixel_size_rad ** 2)
    k = np.arange(p1d.shape[0])
    ell = 2.0 * np.pi * k / pixel_size_rad / 360.0
    return ell, p1d


def main():
    interim_dir = os.path.join(CFG_DATA.output_root, CFG_DATA.intermediate_dirname)
    weights_dir = os.path.join(CFG_DATA.output_root, CFG_DATA.weights_dirname)
    os.makedirs(weights_dir, exist_ok=True)

    manifest = np.load(os.path.join(CFG_DATA.output_root, "manifest.npz"))
    train_ids = manifest["train_ids"]

    fov_mpc = CFG_GRID.fov_mpc
    N = CFG_GRID.lens_recon_resolution
    D_l = lu.angular_diameter_distance_mpc(CFG_LENS.z_lens_assumed,
                                           H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0)
    arcmin_per_mpc = (180.0 * 60.0 / np.pi) / D_l
    pixel_size_arcmin = (2.0 * fov_mpc / N) * arcmin_per_mpc
    pixel_size_rad = pixel_size_arcmin * (np.pi / 180.0 / 60.0)

    mean_beta = lu.mean_beta_over_smail(
        CFG_LENS.z_lens_assumed,
        alpha=CFG_LENS.smail_alpha, beta=CFG_LENS.smail_beta, z0=CFG_LENS.smail_z0,
        z_max=CFG_LENS.smail_z_max, H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0)
    print(f"<beta>={mean_beta:.6f}, pixel={pixel_size_arcmin:.4f} arcmin, "
          f"pixel_rad={pixel_size_rad:.6e}")

    ps_sum = None
    n_ok = 0
    ell_ref = None
    for sid in train_ids:
        f = os.path.join(interim_dir, f"sample_{int(sid):06d}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        kappa_train = (mean_beta * d["kappa_inf"]).astype(np.float32)
        ell, p = measure_ps(kappa_train, pixel_size_rad)
        if ps_sum is None:
            ps_sum = p.astype(np.float64)
            ell_ref = ell
        else:
            ps_sum += p.astype(np.float64)
        n_ok += 1
        if n_ok % 250 == 0:
            print(f"  P(k) accumulated: {n_ok}/{len(train_ids)}")

    if n_ok == 0:
        raise RuntimeError("no training samples found in intermediate/; run stage 1 first")
    ps_mean = ps_sum / n_ok

    out = np.stack([ell_ref.astype(np.float64), ps_mean], axis=0)
    out_path = os.path.join(weights_dir, "cluster_kappa_PS_theory.npy")
    np.save(out_path, out)
    np.save(os.path.join(weights_dir, "mean_beta.npy"), float(mean_beta))
    np.save(os.path.join(weights_dir, "pixel_size_rad.npy"), float(pixel_size_rad))
    print(f"saved {out_path} shape={out.shape}")
    print("saved mean_beta.npy and pixel_size_rad.npy alongside")


if __name__ == "__main__":
    main()
