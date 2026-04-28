#!/usr/bin/env python3
"""
Stage 3: assemble final HDF5 train/test files from per-sample stage-1 npz outputs and
per-sample stage-2b posterior outputs.

Variable resolution:
  --final-image-resolution N   (default 16; lens_recon_resolution must be divisible by N)
  --final-cube-resolution  M   (default 16; interim_cube_resolution must be divisible by M)

This stage avg-pools kappa channels (saved at lens_recon_resolution=128) and the density cube
(saved at interim_cube_resolution=64) to the chosen final resolutions; dynamical channels
(gal_xy, gal_vz_xy, gal_vz_disp_xy) are re-binned at the final resolution from the saved
raw cluster-member catalog. n_gal_pix is downsampled by SUM (preserves count semantics).

Final 8-channel image stack:
    [kappa_E_mean, kappa_E_std,
     gamma1_obs, gamma2_obs,
     n_gal_pix,
     gal_xy, gal_vz_xy, gal_vz_disp_xy]
"""
import os
import sys
import argparse
import h5py
import numpy as np
import multiprocessing as mp
from typing import Dict, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (CFG_GRID, CFG_LENS, CFG_COSMO, CFG_SCALE, CFG_CAT, CFG_DATA)
import lensing_utils as lu


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-image-resolution", type=int, default=CFG_GRID.final_image_resolution)
    ap.add_argument("--final-cube-resolution", type=int, default=CFG_GRID.final_cube_resolution)
    ap.add_argument("--max-nodes", type=int, default=CFG_CAT.max_nodes)
    return ap.parse_args()


def build_dynamical_maps(gal_pos, gal_vel, fov, N, scaling_cfg):
    """Density (counts/Ngal), mean vz, vz dispersion on N x N from raw rotated catalog."""
    if gal_pos.shape[0] == 0:
        z = np.zeros((N, N), dtype=np.float32)
        return z.copy(), z.copy(), z.copy()
    x, y = gal_pos[:, 0], gal_pos[:, 1]
    vz = gal_vel[:, 2].astype(np.float32)

    counts = lu.bin2d_count(x, y, fov, N)
    n_total = max(counts.sum(), 1.0)
    gal_xy = (counts / n_total).astype(np.float32)

    vz_sum = lu.bin2d_sum(x, y, vz, fov, N)
    vz2_sum = lu.bin2d_sum(x, y, vz ** 2, fov, N)

    vz_mean = np.zeros((N, N), dtype=np.float32)
    vz_disp = np.zeros((N, N), dtype=np.float32)
    valid = counts > 0
    vz_mean[valid] = vz_sum[valid] / counts[valid]
    multi = counts > 1
    if multi.any():
        m1 = vz_sum[multi] / counts[multi]
        m2 = vz2_sum[multi] / counts[multi]
        var = np.maximum(m2 - m1 ** 2, 0.0)
        vz_disp[multi] = np.sqrt(var).astype(np.float32)

    vz_mean = (vz_mean - scaling_cfg.vel_mean) / scaling_cfg.vel_std
    vz_disp = vz_disp / scaling_cfg.vel_std
    return gal_xy, vz_mean.astype(np.float32), vz_disp.astype(np.float32)


def xy_to_pixel_coords(x, y, fov, N):
    pix_x = (x + fov) / (2.0 * fov + 1e-30) * (N - 1)
    pix_y = (y + fov) / (2.0 * fov + 1e-30) * (N - 1)
    return np.stack([np.clip(pix_x, 0.0, N - 1),
                     np.clip(pix_y, 0.0, N - 1)], axis=-1).astype(np.float32)


def log_standardize_with_floor(x, mean, std, floor_value):
    out = np.full_like(x, floor_value, dtype=np.float32)
    mask = x > 0.0
    if mask.any():
        v = np.log10(x[mask].astype(np.float64))
        z = (v - mean) / std
        out[mask] = np.maximum(z, floor_value).astype(np.float32)
    return out


def rho_cube_to_mass_msun(cube, fov):
    voxel = (2.0 * fov / cube.shape[0]) ** 3
    return float(cube.sum() * voxel)


def rho_cube_to_axis_lengths_mpc(cube, fov):
    N = cube.shape[0]
    vox = 2.0 * fov / N
    coords = (np.arange(N, dtype=np.float64) + 0.5) * vox - fov
    Z, Y, X = np.meshgrid(coords, coords, coords, indexing="ij")
    mass = cube.astype(np.float64) * vox ** 3
    M = mass.sum()
    if M <= 0:
        return float("nan"), float("nan"), float("nan")
    xc = (mass * X).sum() / M
    yc = (mass * Y).sum() / M
    zc = (mass * Z).sum() / M
    dx, dy, dz = X - xc, Y - yc, Z - zc
    S = np.array([
        [(mass * dx * dx).sum(), (mass * dx * dy).sum(), (mass * dx * dz).sum()],
        [(mass * dy * dx).sum(), (mass * dy * dy).sum(), (mass * dy * dz).sum()],
        [(mass * dz * dx).sum(), (mass * dz * dy).sum(), (mass * dz * dz).sum()],
    ]) / M
    evs = np.sort(np.linalg.eigvalsh(S))[::-1]
    a, b, c = np.sqrt(np.clip(evs, 0.0, None))
    return float(a), float(b), float(c)


def assemble_one(args):
    sid, paths, fov, scaling_cfg, N_img, N_cube, max_nodes = args
    interim_path, posterior_path = paths
    d = np.load(interim_path)
    p = np.load(posterior_path)

    factor_img = CFG_GRID.lens_recon_resolution // N_img
    if CFG_GRID.lens_recon_resolution % N_img != 0:
        raise RuntimeError(
            f"lens_recon_resolution {CFG_GRID.lens_recon_resolution} not divisible by {N_img}")
    factor_cube = CFG_GRID.interim_cube_resolution // N_cube
    if CFG_GRID.interim_cube_resolution % N_cube != 0:
        raise RuntimeError(
            f"interim_cube_resolution {CFG_GRID.interim_cube_resolution} not divisible by {N_cube}")

    kE_mean = lu.avg_pool_2d(p["kappa_E_mean"].astype(np.float32), factor_img)
    kE_std = lu.avg_pool_2d(p["kappa_E_std"].astype(np.float32), factor_img)
    g1_obs = lu.avg_pool_2d(d["gamma1_obs"].astype(np.float32), factor_img)
    g2_obs = lu.avg_pool_2d(d["gamma2_obs"].astype(np.float32), factor_img)
    # downsample n_gal_pix by SUM (avg-pool * factor^2 = sum) so coarse-pixel counts make sense
    n_pix = lu.avg_pool_2d(d["n_gal_pix"].astype(np.float32), factor_img) * (factor_img ** 2)

    gal_pos = d["gal_pos"]
    gal_vel = d["gal_vel"]
    gal_xy, gal_vz_xy, gal_vz_disp_xy = build_dynamical_maps(
        gal_pos, gal_vel, fov, N_img, scaling_cfg)

    density_cube = lu.avg_pool_3d(d["density_cube"].astype(np.float32), factor_cube)

    cube_mass = rho_cube_to_mass_msun(density_cube, fov)
    a_mpc, b_mpc, c_mpc = rho_cube_to_axis_lengths_mpc(density_cube, fov)

    # padded cluster-member galaxy catalog (positions/velocities standardized)
    x_s = (gal_pos[:, 0] - scaling_cfg.pos_mean) / scaling_cfg.pos_std
    y_s = (gal_pos[:, 1] - scaling_cfg.pos_mean) / scaling_cfg.pos_std
    z_s = (gal_pos[:, 2] - scaling_cfg.pos_mean) / scaling_cfg.pos_std
    vx_s = (gal_vel[:, 0] - scaling_cfg.vel_mean) / scaling_cfg.vel_std
    vy_s = (gal_vel[:, 1] - scaling_cfg.vel_mean) / scaling_cfg.vel_std
    vz_s = (gal_vel[:, 2] - scaling_cfg.vel_mean) / scaling_cfg.vel_std
    N_gal = x_s.shape[0]
    n_feat = np.full((N_gal,), float(N_gal), dtype=np.float32)
    feats = np.stack([x_s, y_s, vz_s, n_feat], axis=-1).astype(np.float32)
    targs = np.stack([z_s, vx_s, vy_s], axis=-1).astype(np.float32)
    pix_coords = xy_to_pixel_coords(gal_pos[:, 0], gal_pos[:, 1], fov, N_img)
    Nuse = min(N_gal, max_nodes)
    feat_pad = np.zeros((max_nodes, 4), dtype=np.float32)
    targ_pad = np.zeros((max_nodes, 3), dtype=np.float32)
    pix_pad = np.zeros((max_nodes, 2), dtype=np.float32)
    mask_pad = np.zeros((max_nodes,), dtype=np.float32)
    feat_pad[:Nuse] = feats[:Nuse]
    targ_pad[:Nuse] = targs[:Nuse]
    pix_pad[:Nuse] = pix_coords[:Nuse]
    mask_pad[:Nuse] = 1.0

    return dict(
        sample_id=int(sid),
        cluster_idx=int(d["cluster_idx"]),
        sim=str(d["sim"]),
        proj_vec=d["proj_vec"].astype(np.float32),
        halo_mass_msun=float(d["halo_mass_msun"]),
        n_gal=int(N_gal),

        kE_mean=kE_mean.astype(np.float32),
        kE_std=kE_std.astype(np.float32),
        g1_obs=g1_obs.astype(np.float32),
        g2_obs=g2_obs.astype(np.float32),
        n_pix=n_pix.astype(np.float32),
        gal_xy=gal_xy.astype(np.float32),
        gal_vz_xy=gal_vz_xy.astype(np.float32),
        gal_vz_disp_xy=gal_vz_disp_xy.astype(np.float32),
        density_cube_raw=density_cube.astype(np.float32),

        cube_mass_log10=np.float32(np.log10(max(cube_mass, 1e-30))),
        axis_lengths_mpc=np.array([a_mpc, b_mpc, c_mpc], dtype=np.float32),

        gal_features=feat_pad,
        gal_targets=targ_pad,
        gal_pixel_coords=pix_pad,
        gal_mask=mask_pad,
    )


def compute_stats(samples):
    cube_logs = []
    kE_means, kE_stds, g_obs, n_pixs = [], [], [], []
    cube_mass_logs = []
    axes = []

    for s in samples:
        c = s["density_cube_raw"]
        cm = c > 0
        if cm.any():
            cube_logs.append(np.log10(c[cm].astype(np.float64)))
        kE_means.append(s["kE_mean"].ravel())
        kE_stds.append(s["kE_std"].ravel())
        g_obs.append(s["g1_obs"].ravel()); g_obs.append(s["g2_obs"].ravel())
        n_pixs.append(s["n_pix"].ravel())
        cube_mass_logs.append(float(s["cube_mass_log10"]))
        axes.append(s["axis_lengths_mpc"].astype(np.float64))

    cube_logs = np.concatenate(cube_logs) if cube_logs else np.zeros(0)
    cube_log10_mean = float(cube_logs.mean()) if cube_logs.size else 0.0
    cube_log10_std = float(cube_logs.std() + 1e-6) if cube_logs.size else 1.0

    def stats_of(arrs):
        a = np.concatenate(arrs).astype(np.float64)
        return float(a.mean()), float(a.std() + 1e-6)

    kE_mean_mu, kE_mean_sd = stats_of(kE_means)
    kE_std_mu, kE_std_sd = stats_of(kE_stds)
    g_obs_mu, g_obs_sd = stats_of(g_obs)
    n_pix_mu, n_pix_sd = stats_of(n_pixs)

    cube_mass_logs = np.array(cube_mass_logs, dtype=np.float64)
    axes = np.stack(axes, axis=0)
    return dict(
        cube_log10_mean=cube_log10_mean, cube_log10_std=cube_log10_std,
        kE_mean_mean=kE_mean_mu, kE_mean_std=kE_mean_sd,
        kE_std_mean=kE_std_mu, kE_std_std=kE_std_sd,
        g_obs_mean=g_obs_mu, g_obs_std=g_obs_sd,
        n_pix_mean=n_pix_mu, n_pix_std=n_pix_sd,
        cube_mass_log10_mean=float(cube_mass_logs.mean()),
        cube_mass_log10_std=float(cube_mass_logs.std() + 1e-6),
        axis_mean=axes.mean(axis=0),
        axis_std=axes.std(axis=0) + 1e-6,
        floor_value=CFG_GRID.floor_value,
    )


def write_sample_to_hdf5(h5, idx, s, stats):
    grp = h5.create_group(f"{idx:06d}")
    grp.attrs["id"] = int(idx)
    grp.attrs["sample_id"] = int(s["sample_id"])
    grp.attrs["simulation"] = s["sim"]
    grp.attrs["cluster_index"] = int(s["cluster_idx"])
    grp.attrs["cluster_mass"] = float(s["halo_mass_msun"])
    grp.attrs["n_galaxies"] = int(s["n_gal"])

    cube = log_standardize_with_floor(
        s["density_cube_raw"], stats["cube_log10_mean"], stats["cube_log10_std"],
        stats["floor_value"])

    def zscore(arr, mu, sd):
        return ((arr.astype(np.float64) - mu) / sd).astype(np.float32)

    kE_mean_z = zscore(s["kE_mean"], stats["kE_mean_mean"], stats["kE_mean_std"])
    kE_std_z = zscore(s["kE_std"], stats["kE_std_mean"], stats["kE_std_std"])
    g1_z = zscore(s["g1_obs"], stats["g_obs_mean"], stats["g_obs_std"])
    g2_z = zscore(s["g2_obs"], stats["g_obs_mean"], stats["g_obs_std"])
    n_pix_z = zscore(s["n_pix"], stats["n_pix_mean"], stats["n_pix_std"])

    images = np.stack([
        kE_mean_z, kE_std_z, g1_z, g2_z, n_pix_z,
        s["gal_xy"], s["gal_vz_xy"], s["gal_vz_disp_xy"],
    ], axis=0).astype(np.float32)

    raw_mass_log10 = float(s["cube_mass_log10"])
    raw_axes = s["axis_lengths_mpc"].astype(np.float32)
    std_mass = (raw_mass_log10 - stats["cube_mass_log10_mean"]) / stats["cube_mass_log10_std"]
    std_axes = (raw_axes.astype(np.float64) - stats["axis_mean"]) / stats["axis_std"]
    globals_target = np.concatenate(
        [np.array([std_mass], dtype=np.float32), std_axes.astype(np.float32)], axis=0)

    grp.create_dataset("projection_vector", data=s["proj_vec"])
    grp.create_dataset("images", data=images, compression="gzip")
    grp.create_dataset("density_cube", data=cube, compression="gzip")
    grp.create_dataset("gal_features", data=s["gal_features"], compression="gzip")
    grp.create_dataset("gal_targets", data=s["gal_targets"], compression="gzip")
    grp.create_dataset("gal_pixel_coords", data=s["gal_pixel_coords"], compression="gzip")
    grp.create_dataset("mask", data=s["gal_mask"], compression="gzip")
    grp.create_dataset("cube_mass_log10_msun", data=np.float32(raw_mass_log10))
    grp.create_dataset("axis_lengths_mpc", data=raw_axes)
    grp.create_dataset("globals_target", data=globals_target)


def write_global_attrs(f, fov, N_img, N_cube, max_nodes, stats):
    f.attrs["map_fov_mpc"] = fov
    f.attrs["image_resolution"] = N_img
    f.attrs["cube_resolution"] = N_cube
    f.attrs["lens_recon_resolution"] = CFG_GRID.lens_recon_resolution
    f.attrs["interim_cube_resolution"] = CFG_GRID.interim_cube_resolution
    f.attrs["floor_value"] = CFG_GRID.floor_value
    f.attrs["max_nodes"] = max_nodes

    f.attrs["pos_mean"] = CFG_SCALE.pos_mean
    f.attrs["pos_std"] = CFG_SCALE.pos_std
    f.attrs["vel_mean"] = CFG_SCALE.vel_mean
    f.attrs["vel_std"] = CFG_SCALE.vel_std

    f.attrs["z_lens_assumed"] = CFG_LENS.z_lens_assumed
    f.attrs["sigma_e_per_component"] = CFG_LENS.sigma_e_per_component
    f.attrs["n_source_per_arcmin2"] = CFG_LENS.n_source_per_arcmin2
    f.attrs["smail_alpha"] = CFG_LENS.smail_alpha
    f.attrs["smail_beta"] = CFG_LENS.smail_beta
    f.attrs["smail_z0"] = CFG_LENS.smail_z0
    f.attrs["r_inner_arcsec"] = CFG_LENS.r_inner_arcsec
    f.attrs["r_outer_arcmin"] = CFG_LENS.r_outer_arcmin
    f.attrs["alpha_mag"] = CFG_LENS.alpha_mag
    f.attrs["use_magnification_bias"] = bool(CFG_LENS.use_magnification_bias)

    f.attrs["cube_log10_mean"] = stats["cube_log10_mean"]
    f.attrs["cube_log10_std"] = stats["cube_log10_std"]
    f.attrs["kE_mean_mean"] = stats["kE_mean_mean"]
    f.attrs["kE_mean_std"] = stats["kE_mean_std"]
    f.attrs["kE_std_mean"] = stats["kE_std_mean"]
    f.attrs["kE_std_std"] = stats["kE_std_std"]
    f.attrs["g_obs_mean"] = stats["g_obs_mean"]
    f.attrs["g_obs_std"] = stats["g_obs_std"]
    f.attrs["n_pix_mean"] = stats["n_pix_mean"]
    f.attrs["n_pix_std"] = stats["n_pix_std"]
    f.attrs["cube_mass_log10_mean"] = stats["cube_mass_log10_mean"]
    f.attrs["cube_mass_log10_std"] = stats["cube_mass_log10_std"]
    f.attrs["axis_mean"] = stats["axis_mean"].astype(np.float64)
    f.attrs["axis_std"] = stats["axis_std"].astype(np.float64)

    f.attrs["image_channels"] = np.array(
        [b"kappa_E_mean", b"kappa_E_std",
         b"gamma1_obs", b"gamma2_obs",
         b"n_gal_pix",
         b"gal_xy", b"gal_vz_xy", b"gal_vz_disp_xy"],
        dtype="S20")
    f.attrs["density_cube_order"] = "zyx"
    f.attrs["gal_feature_columns"] = np.array([b"x", b"y", b"vz", b"Ngal"], dtype="S8")
    f.attrs["gal_target_columns"] = np.array([b"z", b"vx", b"vy"], dtype="S8")
    f.attrs["gal_pixel_coords_columns"] = np.array([b"x_pix", b"y_pix"], dtype="S8")
    f.attrs["globals_target_columns"] = np.array(
        [b"mass_log10_msun", b"axis_a_mpc", b"axis_b_mpc", b"axis_c_mpc"], dtype="S20")


def main():
    args = parse_args()
    fov = CFG_GRID.fov_mpc
    N_img = args.final_image_resolution
    N_cube = args.final_cube_resolution

    interim_dir = os.path.join(CFG_DATA.output_root, CFG_DATA.intermediate_dirname)
    posterior_dir = os.path.join(CFG_DATA.output_root, CFG_DATA.posterior_dirname)
    final_dir = os.path.join(CFG_DATA.output_root, CFG_DATA.final_dirname)
    os.makedirs(final_dir, exist_ok=True)

    train_path = os.path.join(final_dir, CFG_DATA.train_h5_template.format(img=N_img, cube=N_cube))
    test_path = os.path.join(final_dir, CFG_DATA.test_h5_template.format(img=N_img, cube=N_cube))
    for pth in (train_path, test_path):
        if os.path.exists(pth):
            os.remove(pth)

    manifest = np.load(os.path.join(CFG_DATA.output_root, "manifest.npz"))
    train_ids = manifest["train_ids"]
    test_ids = manifest["test_ids"]

    def make_jobs(ids):
        out = []
        for sid in ids:
            ip = os.path.join(interim_dir, f"sample_{int(sid):06d}.npz")
            pp = os.path.join(posterior_dir, f"posterior_{int(sid):06d}.npz")
            if os.path.exists(ip) and os.path.exists(pp):
                out.append((int(sid), (ip, pp), fov, CFG_SCALE, N_img, N_cube, args.max_nodes))
        return out

    train_jobs = make_jobs(train_ids)
    test_jobs = make_jobs(test_ids)
    print(f"train: {len(train_jobs)}/{len(train_ids)} ready, "
          f"test: {len(test_jobs)}/{len(test_ids)} ready", flush=True)

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"using {n_workers} workers", flush=True)

    print("assembling train samples...", flush=True)
    train_built = []
    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        for i, s in enumerate(pool.imap_unordered(assemble_one, train_jobs, chunksize=4),
                              start=1):
            train_built.append(s)
            if i % 250 == 0:
                print(f"  built train {i}/{len(train_jobs)}", flush=True)

    print("assembling test samples...", flush=True)
    test_built = []
    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        for i, s in enumerate(pool.imap_unordered(assemble_one, test_jobs, chunksize=4),
                              start=1):
            test_built.append(s)
            if i % 250 == 0:
                print(f"  built test {i}/{len(test_jobs)}", flush=True)

    print("computing global train-set stats...", flush=True)
    stats = compute_stats(train_built)
    for k, v in stats.items():
        if not isinstance(v, np.ndarray):
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print("writing HDF5...", flush=True)
    for path, samples in [(train_path, train_built), (test_path, test_built)]:
        with h5py.File(path, "w") as f:
            write_global_attrs(f, fov, N_img, N_cube, args.max_nodes, stats)
            for idx, s in enumerate(samples):
                write_sample_to_hdf5(f, idx, s, stats)
                if (idx + 1) % 250 == 0:
                    print(f"  wrote {idx + 1}/{len(samples)} -> {os.path.basename(path)}",
                          flush=True)

    print("stage 3 done.")
    print(f"train: {train_path}")
    print(f"test : {test_path}")


if __name__ == "__main__":
    main()
