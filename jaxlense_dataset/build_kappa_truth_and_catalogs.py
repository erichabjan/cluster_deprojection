#!/usr/bin/env python3
"""
Stage 1: per (cluster, sim, projection), build:
  - kappa_inf truth at lens_recon_resolution (default 128) from Sigma / Sigma_crit_inf(z_l)
  - synthetic background-galaxy shape catalog under Bahe+ 2012 sec 3.2 with Smail n(z)
  - binned (gamma1_obs, gamma2_obs) and n_gal_pix at lens_recon_resolution
  - 3D density cube at interim_cube_resolution
  - rotated cluster-member catalog (positions, velocities) for stage-3 dynamical channels

Output:
  {output_root}/{intermediate_dirname}/sample_{NNNNNN}.npz   one per (cluster, sim, projection)
  {output_root}/manifest.npz                                  train/test split by cluster

Run: cl_dyn env, multiple CPUs.
"""
import os
import sys
import numpy as np
import multiprocessing as mp
from typing import Dict, Tuple

dirc_path = "/home/habjan.e/"
sys.path.append(dirc_path + "TNG/TNG_cluster_dynamics")
import TNG_DA  # rotate_to_viewing_frame

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (CFG_GRID, CFG_LENS, CFG_COSMO, CFG_DATA)
import lensing_utils as lu

BOXSIZE_MPC_OVER_H = 400.0   # BAHAMAS box (cMpc/h)
EPS = 1e-9


def _bin3d_sum(x, y, z, w, lim, N):
    H, _ = np.histogramdd(
        np.stack([z, y, x], axis=-1).astype(np.float64),
        bins=(N, N, N),
        range=(lim, lim, lim),
        weights=w.astype(np.float64),
    )
    return H.astype(np.float32)


def project_matter_components(data, proj_vec, h_sim):
    """Return dict of {'dm': (pos_mpc_rot, mass_msun), 'gas': ..., 'star': ..., 'bh': ...}."""
    boxsize = BOXSIZE_MPC_OVER_H
    a_scale = float(data["a"])

    def prep(pos_key, mass_key):
        if pos_key not in data.files or mass_key not in data.files:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        pos_raw = data[pos_key].astype(np.float64)
        mass_raw = data[mass_key].astype(np.float64)
        if pos_raw.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        dif = pos_raw - data["CoP"]
        coords = (dif + 0.5 * boxsize) % boxsize - 0.5 * boxsize
        pos_mpc = (coords / (h_sim * a_scale)) + EPS
        mass_msun = mass_raw / h_sim                                 # Msun/h -> Msun
        rot, _ = TNG_DA.rotate_to_viewing_frame(pos_mpc, np.zeros_like(pos_mpc), proj_vec)
        return rot.astype(np.float32), mass_msun.astype(np.float32)

    return {
        "dm":   prep("dm_pos",   "dm_mass"),
        "gas":  prep("gas_pos",  "gas_mass"),
        "star": prep("star_pos", "star_mass"),
        "bh":   prep("bh_pos",   "bh_mass"),
    }


def build_sigma_2d(components, fov, N):
    """Sum mass per pixel area -> Sigma in Msun/Mpc^2."""
    sigma = np.zeros((N, N), dtype=np.float32)
    for pos, m in components.values():
        if pos.shape[0] == 0:
            continue
        keep = ((pos[:, 0] >= -fov) & (pos[:, 0] <= fov)
                & (pos[:, 1] >= -fov) & (pos[:, 1] <= fov))
        if keep.any():
            sigma += lu.bin2d_sum(pos[keep, 0], pos[keep, 1], m[keep], fov, N)
    pix_area = (2.0 * fov / N) ** 2
    return sigma / (pix_area + 1e-30)


def build_density_cube(components, fov, N):
    cube = np.zeros((N, N, N), dtype=np.float32)
    for pos, m in components.values():
        if pos.shape[0] == 0:
            continue
        keep = ((pos[:, 0] >= -fov) & (pos[:, 0] <= fov)
                & (pos[:, 1] >= -fov) & (pos[:, 1] <= fov)
                & (pos[:, 2] >= -fov) & (pos[:, 2] <= fov))
        if keep.any():
            cube += _bin3d_sum(pos[keep, 0], pos[keep, 1], pos[keep, 2],
                               m[keep], (-fov, fov), N)
    voxel = (2.0 * fov / N) ** 3
    return cube / (voxel + 1e-30)


def build_member_catalog(data, proj_vec, h_sim):
    """Bright cluster-member subhalos rotated into the viewing frame."""
    boxsize = BOXSIZE_MPC_OVER_H
    a_scale = float(data["a"])
    bright = data["sub_massTotal"][:, 4] != 0
    if not bright.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    dif = data["sub_pos"][bright] - data["CoP"]
    coords = (dif + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    pos_mpc = (coords / (h_sim * a_scale)) + EPS
    vel = data["sub_vel"][bright].astype(np.float64)
    vel = vel - vel.mean(axis=0, keepdims=True)
    rot_pos, rot_vel = TNG_DA.rotate_to_viewing_frame(pos_mpc, vel, proj_vec)
    return rot_pos.astype(np.float32), rot_vel.astype(np.float32)


def _empty_catalog():
    return {k: np.zeros((0,), dtype=np.float32)
            for k in ("x", "y", "z_s", "beta", "e1_obs", "e2_obs")}


def draw_shape_catalog(kappa_inf, gamma1_inf, gamma2_inf, fov_mpc, z_l, rng):
    """
    Bahe+ 2012 sec 3.2 with Smail n(z):
      - n_source = 30/arcmin^2 over an annulus [r_inner_arcsec, r_outer_arcmin]
      - per-galaxy z_s ~ Smail
      - magnification thinning with rate ~ mu^(2.5*alpha_mag - 1)
      - reduced shear g = beta*gamma_inf / (1 - beta*kappa_inf)
      - epsilon_obs = (epsilon_int + g) / (1 + g* epsilon_int)
    """
    D_l = lu.angular_diameter_distance_mpc(z_l, H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0)
    arcmin_per_mpc = (180.0 * 60.0 / np.pi) / D_l
    r_in_mpc = (CFG_LENS.r_inner_arcsec / 60.0) / arcmin_per_mpc
    r_out_mpc = CFG_LENS.r_outer_arcmin / arcmin_per_mpc
    r_out_mpc = min(r_out_mpc, fov_mpc * 0.99)              # keep inside the box

    area_arcmin2 = np.pi * (CFG_LENS.r_outer_arcmin ** 2
                            - (CFG_LENS.r_inner_arcsec / 60.0) ** 2)
    n_mean = CFG_LENS.n_source_per_arcmin2 * area_arcmin2

    exponent = 2.5 * CFG_LENS.alpha_mag - 1.0
    weight_max_bound = max(CFG_LENS.mu_max ** exponent,
                           (1.0 / CFG_LENS.mu_max) ** exponent)
    if not CFG_LENS.use_magnification_bias:
        weight_max_bound = 1.0

    n_proposed = int(rng.poisson(n_mean * weight_max_bound * CFG_LENS.oversample_safety))
    if n_proposed == 0:
        return _empty_catalog()

    # rejection-sample positions in the annulus from a uniform square proposal
    cand_x = np.empty(0); cand_y = np.empty(0)
    while cand_x.size < n_proposed:
        chunk = max(n_proposed - cand_x.size, 256)
        rx = rng.uniform(-r_out_mpc, r_out_mpc, size=chunk * 2)
        ry = rng.uniform(-r_out_mpc, r_out_mpc, size=chunk * 2)
        rr = np.sqrt(rx ** 2 + ry ** 2)
        keep = (rr >= r_in_mpc) & (rr <= r_out_mpc)
        cand_x = np.concatenate([cand_x, rx[keep]])
        cand_y = np.concatenate([cand_y, ry[keep]])
    x_g = cand_x[:n_proposed].astype(np.float64)
    y_g = cand_y[:n_proposed].astype(np.float64)

    z_s = lu.sample_smail_redshifts(
        n_proposed,
        alpha=CFG_LENS.smail_alpha, beta=CFG_LENS.smail_beta, z0=CFG_LENS.smail_z0,
        z_max=CFG_LENS.smail_z_max, rng=rng)
    beta = lu.beta_factor(z_l, z_s, H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0)

    k_inf_g = lu.bilinear_sample(kappa_inf, x_g, y_g, fov_mpc)
    g1_inf_g = lu.bilinear_sample(gamma1_inf, x_g, y_g, fov_mpc)
    g2_inf_g = lu.bilinear_sample(gamma2_inf, x_g, y_g, fov_mpc)

    k_eff = beta * k_inf_g
    g1_eff = beta * g1_inf_g
    g2_eff = beta * g2_inf_g

    # magnification mu = 1 / |(1-k)^2 - |gamma|^2|; clipped to avoid critical curves
    denom = (1.0 - k_eff) ** 2 - (g1_eff ** 2 + g2_eff ** 2)
    mu = 1.0 / np.where(np.abs(denom) > 1e-3, denom, np.sign(denom + 1e-30) * 1e-3)
    mu = np.clip(mu, 1.0 / CFG_LENS.mu_max, CFG_LENS.mu_max)

    if CFG_LENS.use_magnification_bias:
        weight = (mu ** exponent) / weight_max_bound
        weight = np.clip(weight, 0.0, 1.0)
        keep_mag = rng.uniform(size=n_proposed) < weight
    else:
        keep_mag = np.ones(n_proposed, dtype=bool)

    x_g, y_g, z_s, beta = x_g[keep_mag], y_g[keep_mag], z_s[keep_mag], beta[keep_mag]
    k_eff = k_eff[keep_mag]; g1_eff = g1_eff[keep_mag]; g2_eff = g2_eff[keep_mag]

    # reduced shear; reject |g| >= 1 (interior of critical curve)
    g_complex = (g1_eff + 1j * g2_eff) / (1.0 - k_eff + 1e-30)
    keep_g = np.abs(g_complex) < 1.0
    x_g, y_g, z_s, beta = x_g[keep_g], y_g[keep_g], z_s[keep_g], beta[keep_g]
    g_complex = g_complex[keep_g]

    # intrinsic ellipticities
    e_int = (rng.normal(0.0, CFG_LENS.sigma_e_per_component, size=g_complex.shape)
             + 1j * rng.normal(0.0, CFG_LENS.sigma_e_per_component, size=g_complex.shape))
    e_obs = (e_int + g_complex) / (1.0 + np.conjugate(g_complex) * e_int)

    return {
        "x": x_g.astype(np.float32),
        "y": y_g.astype(np.float32),
        "z_s": z_s.astype(np.float32),
        "beta": beta.astype(np.float32),
        "e1_obs": e_obs.real.astype(np.float32),
        "e2_obs": e_obs.imag.astype(np.float32),
    }


def build_one_sample(args) -> Dict:
    npz_path, sim, cluster_idx, proj_vec, sample_id, seed = args
    rng = np.random.default_rng(seed)

    data = np.load(npz_path)
    h_sim = float(data["h"])

    fov = CFG_GRID.fov_mpc
    N_lens = CFG_GRID.lens_recon_resolution
    N_cube = CFG_GRID.interim_cube_resolution

    components = project_matter_components(data, proj_vec, h_sim)

    sigma = build_sigma_2d(components, fov, N_lens)
    sigma_crit_inf = lu.sigma_crit_inf_msun_per_mpc2(
        CFG_LENS.z_lens_assumed, H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0)
    kappa_inf = (sigma / sigma_crit_inf).astype(np.float32)

    gamma1_inf, gamma2_inf = lu.ks93_inv_numpy(kappa_inf, np.zeros_like(kappa_inf))
    gamma1_inf = gamma1_inf.astype(np.float32)
    gamma2_inf = gamma2_inf.astype(np.float32)

    cat = draw_shape_catalog(kappa_inf, gamma1_inf, gamma2_inf, fov,
                             CFG_LENS.z_lens_assumed, rng)

    if cat["x"].size > 0:
        gamma1_obs, n_gal_pix = lu.bin2d_mean(cat["x"], cat["y"], cat["e1_obs"], fov, N_lens)
        gamma2_obs, _         = lu.bin2d_mean(cat["x"], cat["y"], cat["e2_obs"], fov, N_lens)
    else:
        gamma1_obs = np.zeros((N_lens, N_lens), dtype=np.float32)
        gamma2_obs = np.zeros((N_lens, N_lens), dtype=np.float32)
        n_gal_pix  = np.zeros((N_lens, N_lens), dtype=np.float32)

    gal_pos, gal_vel = build_member_catalog(data, proj_vec, h_sim)

    density_cube = build_density_cube(components, fov, N_cube)

    return dict(
        sample_id=int(sample_id),
        cluster_idx=int(cluster_idx),
        sim=str(sim),
        halo_mass_msun=float(data["Mfof"]),
        h_sim=float(h_sim),
        scale_factor=float(data["a"]),
        proj_vec=np.asarray(proj_vec, dtype=np.float32),

        kappa_inf=kappa_inf,
        gamma1_obs=gamma1_obs.astype(np.float32),
        gamma2_obs=gamma2_obs.astype(np.float32),
        n_gal_pix=n_gal_pix.astype(np.float32),

        density_cube=density_cube.astype(np.float32),
        gal_pos=gal_pos.astype(np.float32),
        gal_vel=gal_vel.astype(np.float32),

        n_source=int(cat["x"].size),
        n_member=int(gal_pos.shape[0]),
    )


def write_sample(out_dir, sample):
    fname = os.path.join(out_dir, f"sample_{sample['sample_id']:06d}.npz")
    payload = {k: v for k, v in sample.items() if k != "sample_id"}
    np.savez_compressed(fname, **payload)


def main():
    out_root = CFG_DATA.output_root
    interim_dir = os.path.join(out_root, CFG_DATA.intermediate_dirname)
    os.makedirs(interim_dir, exist_ok=True)

    cluster_inds = np.array([f"{i:03d}" for i in range(*CFG_DATA.cluster_index_range)])
    sims = list(CFG_DATA.simulations)
    n_clusters_total = len(cluster_inds) * len(sims)
    n_proj = max(int(CFG_DATA.dataset_size / n_clusters_total), 1)
    actual = n_clusters_total * n_proj
    print(f"clusters={len(cluster_inds)}, sims={len(sims)}, proj/cluster={n_proj}, "
          f"total samples={actual}")

    rng = np.random.default_rng(CFG_DATA.rng_seed)
    test_size = int(len(cluster_inds) * CFG_DATA.test_fraction)
    test_clusters = set(rng.choice(cluster_inds, size=test_size, replace=False).tolist())

    jobs = []
    sid = 0
    for ci in cluster_inds:
        for sim in sims:
            npz_path = os.path.join(CFG_DATA.sim_data_root, sim, f"GrNm_{ci}.npz")
            for _ in range(n_proj):
                pv = rng.uniform(-1.0, 1.0, size=3)
                pv = pv / max(np.linalg.norm(pv), 1e-12)
                jobs.append((npz_path, sim, int(ci), pv, sid,
                             int(rng.integers(0, 2 ** 31 - 1))))
                sid += 1

    train_ids, test_ids = [], []
    for npz_path, sim, ci, _, sample_id, _ in jobs:
        target = test_ids if f"{ci:03d}" in test_clusters else train_ids
        target.append(sample_id)

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"using {n_workers} workers; starting build...")

    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        for i, sample in enumerate(pool.imap_unordered(build_one_sample, jobs, chunksize=4),
                                   start=1):
            write_sample(interim_dir, sample)
            if i % 250 == 0:
                print(f"  built/wrote {i}/{len(jobs)}", flush=True)

    np.savez(os.path.join(out_root, "manifest.npz"),
             train_ids=np.array(train_ids, dtype=np.int64),
             test_ids=np.array(test_ids, dtype=np.int64),
             n_total=np.int64(len(jobs)))
    print(f"wrote manifest: {len(train_ids)} train, {len(test_ids)} test")
    print("stage 1 done.")


if __name__ == "__main__":
    main()
