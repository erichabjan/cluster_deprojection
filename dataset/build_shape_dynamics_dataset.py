#!/usr/bin/env python3
"""
Build the shape_dynamics dataset for the conditional diffusion model.

Per (cluster, sim, projection) sample, all in ONE viewing frame (a single
projection vector rotates both the cluster members and the matter particles):
  - dynamics conditioning images (3, H, W): gal_xy, gal_vz_xy, gal_vz_disp_xy
  - cluster-member point cloud, padded to max_nodes (features/targets/pixel coords/mask)
  - 3D density target cube (log-standardized with train-set stats) + global targets
  - NEW: weak-lensing source-galaxy point cloud replacing the projected mass image.
    A lens redshift z_l ~ U[z_lens_min, z_lens_max] is drawn per sample; the rotated
    matter particles give Sigma -> kappa_inf -> gamma (KS93) on a
    (shape_grid_size x shape_grid_size) grid over [-fov_mpc, +fov_mpc]; then
    draw_shape_catalog draws sources (LoVoCCS n(z), n_source_per_arcmin2, intrinsic
    ellipticity, magnification bias) and shears them.

    Each sample stores, ragged (no padding, variable N):
      src_features (N, 7) float32: e1, e2, ix, iy, z_s, z_lens, cell_frac
        - ix, iy: integer cell coordinates on the shape grid (stored as float32)
        - cell_frac: (# sources in that cell) / N
      src_cell_id  (N,)   int32: iy * shape_grid_size + ix; rows are PRE-SORTED by this
      src_cell_ptr (S*S+1,) int64: CSR pointers, cell c's sources are rows ptr[c]:ptr[c+1]

Prerequisite: build_lovoccs_redshift_template.py (run once) when
CFG_LENS.empirical_template = 'LoVoCCS'.

Run: cl_dyn env, multiple CPUs (build_dataset.slurm).
"""
import os
import sys
import h5py
import numpy as np
import multiprocessing as mp
from typing import Dict, Tuple

dirc_path = "/home/habjan.e/"
sys.path.append(dirc_path + "TNG/TNG_cluster_dynamics")
import TNG_DA  # rotate_to_viewing_frame

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CFG_GRID, CFG_SCALE, CFG_LENS, CFG_COSMO, CFG_DATA
import lensing_utils as lu

# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------
OUT_DIR = "/projects/mccleary_group/habjan.e/TNG/Data/shape_dynamics/"
TRAIN_FILENAME = "shape_dynamics_train.h5"
VAL_FILENAME = "shape_dynamics_val.h5"

BOXSIZE_MPC_OVER_H = 400.0   # BAHAMAS box (cMpc/h)
EPS = 1e-9

# Populated by _init_worker: empirical n(z) template, foreground-cut above z_lens_max.
_EMPIRICAL_Z_TEMPLATE = None


def _init_worker(z_template):
    global _EMPIRICAL_Z_TEMPLATE
    _EMPIRICAL_Z_TEMPLATE = z_template


# ============================================================
# Dynamics helpers (from conditional_diffusion_data.py)
# ============================================================

def _bin3d_sum(x, y, z, w, lim, N):
    H, _ = np.histogramdd(
        np.stack([z, y, x], axis=-1).astype(np.float64),
        bins=(N, N, N),
        range=(lim, lim, lim),
        weights=w.astype(np.float64),
    )
    return H.astype(np.float32)


def make_galaxy_map_xy(x, y, fov, N_img):
    n = max(len(x), 1)
    w = np.full_like(x, 1.0 / n, dtype=np.float32)
    return lu.bin2d_sum(x, y, w, fov, N_img)


def make_galaxy_vz_mean_xy(x, y, vz, fov, N_img):
    vz_sum = lu.bin2d_sum(x, y, vz.astype(np.float32), fov, N_img)
    count = lu.bin2d_count(x, y, fov, N_img)
    mean_vz = np.zeros_like(vz_sum, dtype=np.float32)
    mask = count > 0
    mean_vz[mask] = vz_sum[mask] / count[mask]
    return mean_vz


def make_galaxy_vz_disp_xy(x, y, vz, fov, N_img):
    """Per-pixel LOS velocity dispersion; empty and 1-galaxy pixels are 0."""
    vz = vz.astype(np.float32)
    vz_sum = lu.bin2d_sum(x, y, vz, fov, N_img)
    vz2_sum = lu.bin2d_sum(x, y, vz ** 2, fov, N_img)
    count = lu.bin2d_count(x, y, fov, N_img)
    disp = np.zeros_like(vz_sum, dtype=np.float32)
    mask = count > 1
    if np.any(mask):
        mean = vz_sum[mask] / count[mask]
        mean2 = vz2_sum[mask] / count[mask]
        disp[mask] = np.sqrt(np.maximum(mean2 - mean ** 2, 0.0)).astype(np.float32)
    return disp


def build_sigma_2d(components, fov, N):
    """Sum mass per pixel area over all matter components -> Sigma in Msun/Mpc^2."""
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


def log_standardize_with_floor(x, mean, std, floor_value, eps=0.0):
    out = np.full_like(x, floor_value, dtype=np.float32)
    mask = x > eps
    if np.any(mask):
        v = np.log10(x[mask].astype(np.float64))
        z = (v - mean) / std
        z = np.maximum(z, floor_value)
        out[mask] = z.astype(np.float32)
    return out


def xy_to_pixel_coords(x, y, fov, N_img):
    x01 = np.clip((x + fov) / (2.0 * fov + CFG_GRID.eps), 0.0, 1.0)
    y01 = np.clip((y + fov) / (2.0 * fov + CFG_GRID.eps), 0.0, 1.0)
    return np.stack([x01 * (N_img - 1), y01 * (N_img - 1)], axis=-1).astype(np.float32)


def rho_cube_to_mass_msun(rho_cube, fov_mpc):
    """rho_cube: (Z,Y,X) in Msun/Mpc^3 -> enclosed mass in Msun."""
    N = rho_cube.shape[0]
    voxel_vol = ((2.0 * fov_mpc) / N) ** 3
    return float(np.sum(rho_cube.astype(np.float64)) * voxel_vol)


def rho_cube_to_axis_lengths_mpc(rho_cube, fov_mpc) -> Tuple[float, float, float]:
    """Mass-weighted shape-tensor axis lengths (a, b, c) in Mpc, largest first."""
    N = rho_cube.shape[0]
    voxel_size = (2.0 * fov_mpc) / N
    mass = rho_cube.astype(np.float64) * voxel_size ** 3
    total_mass = np.sum(mass)
    if total_mass <= 0:
        return float("nan"), float("nan"), float("nan")

    coords_1d = (np.arange(N, dtype=np.float64) + 0.5) * voxel_size - fov_mpc
    z, y, x = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    x_com = np.sum(mass * x) / total_mass
    y_com = np.sum(mass * y) / total_mass
    z_com = np.sum(mass * z) / total_mass
    dx, dy, dz = x - x_com, y - y_com, z - z_com

    S = np.array([
        [np.sum(mass * dx * dx), np.sum(mass * dx * dy), np.sum(mass * dx * dz)],
        [np.sum(mass * dx * dy), np.sum(mass * dy * dy), np.sum(mass * dy * dz)],
        [np.sum(mass * dx * dz), np.sum(mass * dy * dz), np.sum(mass * dz * dz)],
    ], dtype=np.float64) / total_mass

    evals = np.sort(np.linalg.eigvalsh(S))[::-1]
    a, b, c = np.sqrt(np.clip(evals, 0.0, None))
    return float(a), float(b), float(c)


# ============================================================
# Weak-lensing source catalog (from jaxlense_dataset)
# ============================================================

def _empty_catalog():
    return {k: np.zeros((0,), dtype=np.float32)
            for k in ("x", "y", "z_s", "e1_obs", "e2_obs")}


def draw_shape_catalog(kappa_inf, gamma1_inf, gamma2_inf, sigma_crit_inf,
                       fov_mpc, z_l, rng):
    """
    Synthetic source catalog:
      - n_source = CFG_LENS.n_source_per_arcmin2 over the full rectangular FoV
      - remove sources inside r_inner_arcsec to avoid strongly lensed/core region
      - per-galaxy z_s bootstrapped from the empirical n(z) template
      - optional magnification thinning with rate ~ mu^(2.5*alpha_mag - 1)
      - per-source convergence/shear use the per-source critical surface density
            Sigma_crit(z_l, z_s) = c^2 / (4 pi G) * D_s / (D_l * D_ls),
        equivalently kappa_eff(x_i) = Sigma(x_i) / Sigma_crit(z_l, z_{s,i}).
      - reduced shear g = gamma_eff / (1 - kappa_eff)
      - epsilon_obs = (epsilon_int + g) / (1 + conj(g)*epsilon_int)
    """
    if _EMPIRICAL_Z_TEMPLATE is None:
        raise RuntimeError(
            "draw_shape_catalog: worker has no empirical n(z) template "
            "(Pool initializer was not run)."
        )

    D_l = lu.angular_diameter_distance_mpc(
        z_l, H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0
    )
    arcmin_per_mpc = (180.0 * 60.0 / np.pi) / D_l

    # Inner circular exclusion region in physical Mpc.
    r_in_mpc = (CFG_LENS.r_inner_arcsec / 60.0) / arcmin_per_mpc

    # Expected number of sources over the rectangular FoV,
    # excluding the inner strong-lensing/core region.
    field_side_arcmin = 2.0 * fov_mpc * arcmin_per_mpc
    field_area_arcmin2 = field_side_arcmin ** 2
    inner_area_arcmin2 = np.pi * (CFG_LENS.r_inner_arcsec / 60.0) ** 2
    usable_area_arcmin2 = max(field_area_arcmin2 - inner_area_arcmin2, 0.0)

    n_mean = CFG_LENS.n_source_per_arcmin2 * usable_area_arcmin2

    exponent = 2.5 * CFG_LENS.alpha_mag - 1.0
    weight_max_bound = max(
        CFG_LENS.mu_max ** exponent,
        (1.0 / CFG_LENS.mu_max) ** exponent,
    )
    if not CFG_LENS.use_magnification_bias:
        weight_max_bound = 1.0

    n_proposed = int(rng.poisson(n_mean * weight_max_bound))
    if n_proposed == 0:
        return _empty_catalog()

    # Draw sources uniformly over the rectangular FoV,
    # rejecting only the inner core.
    cand_x = np.empty(0, dtype=np.float64)
    cand_y = np.empty(0, dtype=np.float64)

    while cand_x.size < n_proposed:
        chunk = max(n_proposed - cand_x.size, 256)

        rx = rng.uniform(-fov_mpc, fov_mpc, size=chunk * 2)
        ry = rng.uniform(-fov_mpc, fov_mpc, size=chunk * 2)

        rr = np.sqrt(rx ** 2 + ry ** 2)
        keep = rr >= r_in_mpc

        cand_x = np.concatenate([cand_x, rx[keep]])
        cand_y = np.concatenate([cand_y, ry[keep]])

    x_g = cand_x[:n_proposed].astype(np.float64)
    y_g = cand_y[:n_proposed].astype(np.float64)

    z_s = lu.sample_empirical_redshifts(n_proposed, _EMPIRICAL_Z_TEMPLATE, rng=rng)

    # Per-source critical surface density Sigma_crit(z_l, z_s); foreground
    # sources get Sigma_crit -> +inf and thus feel no lensing.
    sigma_crit_src = lu.sigma_crit_msun_per_mpc2(
        z_l, z_s, H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0)

    k_inf_g = lu.bilinear_sample(kappa_inf, x_g, y_g, fov_mpc)
    g1_inf_g = lu.bilinear_sample(gamma1_inf, x_g, y_g, fov_mpc)
    g2_inf_g = lu.bilinear_sample(gamma2_inf, x_g, y_g, fov_mpc)

    # kappa_eff(x_i) = Sigma(x_i) / Sigma_crit(z_l, z_{s,i})
    #               = (Sigma_crit_inf / Sigma_crit(z_l, z_{s,i})) * kappa_inf(x_i),
    # and the same rescaling applies to gamma_inf (KS is linear in Sigma).
    k_eff = (sigma_crit_inf * k_inf_g) / sigma_crit_src
    g1_eff = (sigma_crit_inf * g1_inf_g) / sigma_crit_src
    g2_eff = (sigma_crit_inf * g2_inf_g) / sigma_crit_src

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

    x_g, y_g, z_s = x_g[keep_mag], y_g[keep_mag], z_s[keep_mag]
    k_eff = k_eff[keep_mag]; g1_eff = g1_eff[keep_mag]; g2_eff = g2_eff[keep_mag]

    # reduced shear; reject |g| >= 1 (interior of critical curve)
    g_complex = (g1_eff + 1j * g2_eff) / (1.0 - k_eff + 1e-30)
    keep_g = np.abs(g_complex) < 1.0
    x_g, y_g, z_s = x_g[keep_g], y_g[keep_g], z_s[keep_g]
    g_complex = g_complex[keep_g]

    # intrinsic ellipticities
    e_int = (rng.normal(0.0, CFG_LENS.sigma_e_per_component, size=g_complex.shape)
             + 1j * rng.normal(0.0, CFG_LENS.sigma_e_per_component, size=g_complex.shape))
    e_obs = (e_int + g_complex) / (1.0 + np.conjugate(g_complex) * e_int)

    return {
        "x": x_g.astype(np.float32),
        "y": y_g.astype(np.float32),
        "z_s": z_s.astype(np.float32),
        "e1_obs": e_obs.real.astype(np.float32),
        "e2_obs": e_obs.imag.astype(np.float32),
    }


def process_source_catalog(cat, z_lens, fov, S):
    """
    Map the raw catalog to the model-facing point cloud, doing as much work
    as possible at dataset-creation time:
      - integer cell coords ix, iy on the (S x S) grid over [-fov, fov]
      - cell_frac = (# sources in the cell) / N
      - rows sorted by cell_id = iy*S + ix, with CSR pointers so that
        cell c's sources are rows cell_ptr[c]:cell_ptr[c+1]

    Returns (src_features (N,7) f32, src_cell_id (N,) i32, src_cell_ptr (S*S+1,) i64).
    Feature columns: e1, e2, ix, iy, z_s, z_lens, cell_frac.
    """
    N = cat["x"].size
    if N == 0:
        return (np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((S * S + 1,), dtype=np.int64))

    ix = np.clip(((cat["x"] + fov) / (2.0 * fov) * S).astype(np.int64), 0, S - 1)
    iy = np.clip(((cat["y"] + fov) / (2.0 * fov) * S).astype(np.int64), 0, S - 1)
    cell_id = iy * S + ix

    order = np.argsort(cell_id, kind="stable")
    cell_id = cell_id[order]
    ix, iy = ix[order], iy[order]
    e1, e2 = cat["e1_obs"][order], cat["e2_obs"][order]
    z_s = cat["z_s"][order]

    counts = np.bincount(cell_id, minlength=S * S)
    cell_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    cell_frac = (counts[cell_id] / float(N)).astype(np.float32)

    src_features = np.stack([
        e1.astype(np.float32),
        e2.astype(np.float32),
        ix.astype(np.float32),
        iy.astype(np.float32),
        z_s.astype(np.float32),
        np.full(N, z_lens, dtype=np.float32),
        cell_frac,
    ], axis=-1)

    return src_features, cell_id.astype(np.int32), cell_ptr


# ============================================================
# Sample builder (one viewing frame per sample)
# ============================================================

def build_one_sample(args) -> Dict:
    npz_path, sim, cluster_idx, proj_vec, seed = args
    rng = np.random.default_rng(seed)

    # Per-sample lens redshift, uniform over [z_lens_min, z_lens_max].
    z_lens = float(rng.uniform(CFG_LENS.z_lens_min, CFG_LENS.z_lens_max))

    data = np.load(npz_path)
    h_sim = float(data["h"])
    a_scale = float(data["a"])
    boxsize = BOXSIZE_MPC_OVER_H
    fov = CFG_GRID.fov_mpc

    # -------------------------------
    # Cluster-member galaxies -> viewing frame
    # -------------------------------
    bright = data["sub_massTotal"][:, 4] != 0
    difpos = data["sub_pos"][bright] - data["CoP"]
    coords = (difpos + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    pos = (coords / (h_sim * a_scale)) + EPS  # Mpc

    vel = data["sub_vel"][bright]
    vel = vel - vel.mean(axis=0, keepdims=True)

    ro_pos, ro_vel = TNG_DA.rotate_to_viewing_frame(pos, vel, proj_vec)
    x, y, z = (ro_pos[:, i].astype(np.float32) for i in range(3))
    vx, vy, vz = (ro_vel[:, i].astype(np.float32) for i in range(3))

    # -------------------------------
    # Matter components -> the SAME viewing frame
    # -------------------------------
    def prep_component(pos_key, mass_key):
        if pos_key not in data.files or mass_key not in data.files:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        pos_raw = data[pos_key].astype(np.float64)
        mass_raw = data[mass_key].astype(np.float64)
        if pos_raw.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        dif = pos_raw - data["CoP"]
        cc = (dif + 0.5 * boxsize) % boxsize - 0.5 * boxsize
        pos_mpc = (cc / (h_sim * a_scale)) + EPS
        mass_msun = mass_raw / h_sim                                 # Msun/h -> Msun
        rot, _ = TNG_DA.rotate_to_viewing_frame(pos_mpc, np.zeros_like(pos_mpc), proj_vec)
        return rot.astype(np.float32), mass_msun.astype(np.float32)

    components = {
        "dm":   prep_component("dm_pos",   "dm_mass"),
        "gas":  prep_component("gas_pos",  "gas_mass"),
        "star": prep_component("star_pos", "star_mass"),
        "bh":   prep_component("bh_pos",   "bh_mass"),
    }

    # -------------------------------
    # Dynamics conditioning images (mass_xy replaced by the source catalog)
    # -------------------------------
    N_img = CFG_GRID.image_resolution
    gal_xy = make_galaxy_map_xy(x, y, fov, N_img)
    gal_vz_xy = (make_galaxy_vz_mean_xy(x, y, vz, fov, N_img)
                 - CFG_SCALE.vel_mean) / CFG_SCALE.vel_std
    gal_vz_disp_xy = make_galaxy_vz_disp_xy(x, y, vz, fov, N_img) / CFG_SCALE.vel_std

    # -------------------------------
    # 3D target cube + global targets
    # -------------------------------
    density_cube = build_density_cube(components, fov, CFG_GRID.cube_resolution)
    cube_mass_msun = rho_cube_to_mass_msun(density_cube, fov)
    cube_mass_log10_msun = np.log10(max(cube_mass_msun, 1e-30))
    axis_lengths_mpc = np.array(rho_cube_to_axis_lengths_mpc(density_cube, fov),
                                dtype=np.float32)

    # -------------------------------
    # Cluster-member point cloud (padded)
    # -------------------------------
    gal_pixel_coords = xy_to_pixel_coords(x, y, fov, N_img)

    x_s = (x - CFG_SCALE.pos_mean) / CFG_SCALE.pos_std
    y_s = (y - CFG_SCALE.pos_mean) / CFG_SCALE.pos_std
    z_sc = (z - CFG_SCALE.pos_mean) / CFG_SCALE.pos_std
    vx_s = (vx - CFG_SCALE.vel_mean) / CFG_SCALE.vel_std
    vy_s = (vy - CFG_SCALE.vel_mean) / CFG_SCALE.vel_std
    vz_s = (vz - CFG_SCALE.vel_mean) / CFG_SCALE.vel_std

    N_gal = x.shape[0]
    n_feat = np.full((N_gal,), float(N_gal), dtype=np.float32)
    gal_features = np.stack([x_s, y_s, vz_s, n_feat], axis=-1).astype(np.float32)
    gal_targets = np.stack([z_sc, vx_s, vy_s], axis=-1).astype(np.float32)

    max_nodes = CFG_DATA.max_nodes
    n_keep = min(N_gal, max_nodes)
    feat_pad = np.zeros((max_nodes, 4), dtype=np.float32)
    targ_pad = np.zeros((max_nodes, 3), dtype=np.float32)
    pix_pad = np.zeros((max_nodes, 2), dtype=np.float32)
    mask = np.zeros((max_nodes,), dtype=np.float32)
    feat_pad[:n_keep] = gal_features[:n_keep]
    targ_pad[:n_keep] = gal_targets[:n_keep]
    pix_pad[:n_keep] = gal_pixel_coords[:n_keep]
    mask[:n_keep] = 1.0

    # -------------------------------
    # Weak-lensing source-galaxy point cloud
    # -------------------------------
    S = CFG_LENS.shape_grid_size
    sigma = build_sigma_2d(components, fov, S)
    sigma_crit_inf = lu.sigma_crit_inf_msun_per_mpc2(
        z_lens, H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0)
    kappa_inf = (sigma / sigma_crit_inf).astype(np.float32)
    gamma1_inf, gamma2_inf = lu.ks93_inv_numpy(kappa_inf, np.zeros_like(kappa_inf))

    cat = draw_shape_catalog(kappa_inf,
                             gamma1_inf.astype(np.float32),
                             gamma2_inf.astype(np.float32),
                             sigma_crit_inf, fov, z_lens, rng)
    src_features, src_cell_id, src_cell_ptr = process_source_catalog(
        cat, z_lens, fov, S)

    return dict(
        sim=str(sim),
        cluster_idx=np.int32(cluster_idx),
        halo_mass=np.float32(data["Mfof"]),
        proj_vec=np.asarray(proj_vec, dtype=np.float32),
        z_lens=np.float32(z_lens),

        gal_xy=gal_xy.astype(np.float32),
        gal_vz_xy=gal_vz_xy.astype(np.float32),
        gal_vz_disp_xy=gal_vz_disp_xy.astype(np.float32),

        gal_features=feat_pad,
        gal_targets=targ_pad,
        gal_pixel_coords=pix_pad,
        mask=mask,
        n_gal=np.int32(N_gal),

        src_features=src_features,
        src_cell_id=src_cell_id,
        src_cell_ptr=src_cell_ptr,
        n_src=np.int32(src_features.shape[0]),

        raw_density_cube=density_cube,
        cube_mass_log10_msun=np.float32(cube_mass_log10_msun),
        axis_lengths_mpc=axis_lengths_mpc,
    )


# ============================================================
# Writing
# ============================================================

def write_static_attrs(f: h5py.File):
    f.attrs["map_fov_mpc"] = CFG_GRID.fov_mpc
    f.attrs["image_resolution"] = CFG_GRID.image_resolution
    f.attrs["cube_resolution"] = CFG_GRID.cube_resolution
    f.attrs["vz_max_kms"] = CFG_GRID.vz_max_kms
    f.attrs["floor_value"] = CFG_GRID.floor_value
    f.attrs["max_nodes"] = CFG_DATA.max_nodes

    f.attrs["pos_mean"] = CFG_SCALE.pos_mean
    f.attrs["pos_std"] = CFG_SCALE.pos_std
    f.attrs["vel_mean"] = CFG_SCALE.vel_mean
    f.attrs["vel_std"] = CFG_SCALE.vel_std
    f.attrs["h"] = CFG_COSMO.h_sim

    # Weak-lensing source-catalog metadata (model-facing).
    f.attrs["shape_grid_size"] = CFG_LENS.shape_grid_size
    f.attrs["shape_fov_mpc"] = CFG_GRID.fov_mpc
    f.attrs["z_lens_min"] = CFG_LENS.z_lens_min
    f.attrs["z_lens_max"] = CFG_LENS.z_lens_max
    f.attrs["n_source_per_arcmin2"] = CFG_LENS.n_source_per_arcmin2
    f.attrs["sigma_e_per_component"] = CFG_LENS.sigma_e_per_component
    f.attrs["empirical_template"] = CFG_LENS.empirical_template
    f.attrs["empirical_redshift_npz_path"] = CFG_LENS.empirical_redshift_npz_path
    f.attrs["src_feature_columns"] = np.array(
        [b"e1", b"e2", b"ix", b"iy", b"z_s", b"z_lens", b"cell_frac"], dtype="S12")

    f.attrs["image_channels"] = np.array(
        [b"gal_xy", b"gal_vz_xy", b"gal_vz_disp_xy"], dtype="S20")
    f.attrs["density_cube_order"] = "zyx"
    f.attrs["gal_feature_columns"] = np.array([b"x", b"y", b"vz", b"Ngal"], dtype="S8")
    f.attrs["gal_target_columns"] = np.array([b"z", b"vx", b"vy"], dtype="S8")
    f.attrs["gal_pixel_coords_columns"] = np.array([b"x_pix", b"y_pix"], dtype="S8")
    f.attrs["globals_target_columns"] = np.array(
        [b"mass_log10_msun", b"axis_a_mpc", b"axis_b_mpc", b"axis_c_mpc"], dtype="S20")


def write_sample_streaming(f: h5py.File, sample_id: int, sample: Dict):
    """Write everything that does not depend on train-set statistics."""
    grp = f.create_group(f"{sample_id:06d}")
    grp.attrs["id"] = int(sample_id)
    grp.attrs["simulation"] = sample["sim"]
    grp.attrs["cluster_index"] = int(sample["cluster_idx"])
    grp.attrs["cluster_mass"] = float(sample["halo_mass"])
    grp.attrs["n_galaxies"] = int(sample["n_gal"])
    grp.attrs["n_sources"] = int(sample["n_src"])
    grp.attrs["z_lens"] = float(sample["z_lens"])

    images = np.stack(
        [sample["gal_xy"], sample["gal_vz_xy"], sample["gal_vz_disp_xy"]],
        axis=0,
    ).astype(np.float32)

    grp.create_dataset("projection_vector", data=sample["proj_vec"])
    grp.create_dataset("images", data=images, compression="gzip")               # (3,H,W)
    grp.create_dataset("gal_features", data=sample["gal_features"], compression="gzip")
    grp.create_dataset("gal_targets", data=sample["gal_targets"], compression="gzip")
    grp.create_dataset("gal_pixel_coords", data=sample["gal_pixel_coords"], compression="gzip")
    grp.create_dataset("mask", data=sample["mask"], compression="gzip")

    grp.create_dataset("src_features", data=sample["src_features"], compression="gzip")
    grp.create_dataset("src_cell_id", data=sample["src_cell_id"], compression="gzip")
    grp.create_dataset("src_cell_ptr", data=sample["src_cell_ptr"], compression="gzip")


def write_sample_targets(f: h5py.File, sample_id: int, cube, mass_log10, axes,
                         cube_mean, cube_std, mass_mean, mass_std, axis_mean, axis_std):
    """Second pass: standardized cube + global targets into the existing group."""
    grp = f[f"{sample_id:06d}"]
    cube_norm = log_standardize_with_floor(
        cube, cube_mean, cube_std, floor_value=CFG_GRID.floor_value)

    std_mass = (float(mass_log10) - mass_mean) / mass_std
    std_axes = (axes.astype(np.float64) - axis_mean) / axis_std
    globals_target = np.concatenate(
        [np.array([std_mass], dtype=np.float32), std_axes.astype(np.float32)], axis=0)

    grp.create_dataset("density_cube", data=cube_norm, compression="gzip")      # (Z,Y,X)
    grp.create_dataset("cube_mass_log10_msun", data=np.float32(mass_log10))
    grp.create_dataset("axis_lengths_mpc", data=axes.astype(np.float32))
    grp.create_dataset("globals_target", data=globals_target)


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_path = os.path.join(OUT_DIR, TRAIN_FILENAME)
    val_path = os.path.join(OUT_DIR, VAL_FILENAME)
    for p in (train_path, val_path):
        if os.path.exists(p):
            os.remove(p)

    # Empirical n(z) template, cut above z_lens_max so every source is
    # background for any per-sample lens redshift.
    z_raw = np.load(CFG_LENS.empirical_redshift_npz_path)[CFG_LENS.empirical_redshift_key]
    z_template = np.asarray(z_raw, dtype=np.float64)
    z_template = z_template[z_template > CFG_LENS.z_lens_max]
    if z_template.size == 0:
        raise ValueError(
            f"empirical redshift template at {CFG_LENS.empirical_redshift_npz_path} "
            f"(key={CFG_LENS.empirical_redshift_key!r}) has no entries above "
            f"z_lens_max={CFG_LENS.z_lens_max}"
        )
    print(f"empirical n(z) [{CFG_LENS.empirical_template}]: {z_template.size} sources, "
          f"<z>={z_template.mean():.3f}, min={z_template.min():.3f}, "
          f"max={z_template.max():.3f}")
    print(f"lens redshift: z_l ~ U[{CFG_LENS.z_lens_min}, {CFG_LENS.z_lens_max}] per sample; "
          f"n_source={CFG_LENS.n_source_per_arcmin2}/arcmin^2, "
          f"shape grid {CFG_LENS.shape_grid_size}x{CFG_LENS.shape_grid_size} "
          f"over +/-{CFG_GRID.fov_mpc} Mpc")

    cluster_inds = np.array([f"{i:03d}" for i in range(*CFG_DATA.cluster_index_range)])
    sims = list(CFG_DATA.simulations)
    n_clusters_total = len(cluster_inds) * len(sims)
    n_proj = max(int(CFG_DATA.dataset_size / n_clusters_total), 1)
    print(f"clusters={len(cluster_inds)}, sims={len(sims)}, proj/cluster={n_proj}, "
          f"total samples={n_clusters_total * n_proj}")

    rng = np.random.default_rng(CFG_DATA.rng_seed)
    test_size = int(len(cluster_inds) * CFG_DATA.test_fraction)
    val_clusters = set(rng.choice(cluster_inds, size=test_size, replace=False).tolist())

    jobs = []
    for ci in cluster_inds:
        for sim in sims:
            npz_path = os.path.join(CFG_DATA.sim_data_root, sim, f"GrNm_{ci}.npz")
            for _ in range(n_proj):
                pv = rng.uniform(-1.0, 1.0, size=3)
                pv = pv / max(np.linalg.norm(pv), 1e-12)
                jobs.append((npz_path, sim, int(ci), pv,
                             int(rng.integers(0, 2 ** 31 - 1))))

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"using {n_workers} workers; streaming samples to h5...")

    # Big ragged arrays are written the moment they arrive; only the small
    # cubes + global targets are held back until train-set stats exist.
    held_targets = {"train": {}, "val": {}}   # sid -> (cube, mass_log10, axes)
    sid_counter = {"train": 0, "val": 0}
    n_src_seen = []

    with h5py.File(train_path, "w") as f_train, h5py.File(val_path, "w") as f_val:
        write_static_attrs(f_train)
        write_static_attrs(f_val)
        files = {"train": f_train, "val": f_val}

        with mp.get_context("spawn").Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(z_template,),
        ) as pool:
            for i, sample in enumerate(
                    pool.imap_unordered(build_one_sample, jobs, chunksize=1), start=1):
                key = ("val" if f"{int(sample['cluster_idx']):03d}" in val_clusters
                       else "train")
                sid = sid_counter[key]
                sid_counter[key] += 1

                write_sample_streaming(files[key], sid, sample)
                held_targets[key][sid] = (
                    sample["raw_density_cube"],
                    float(sample["cube_mass_log10_msun"]),
                    np.asarray(sample["axis_lengths_mpc"]),
                )
                n_src_seen.append(int(sample["n_src"]))

                if i % 250 == 0:
                    print(f"  built/wrote {i}/{len(jobs)} "
                          f"(train={sid_counter['train']}, val={sid_counter['val']})",
                          flush=True)

        n_src_seen = np.asarray(n_src_seen)
        print(f"source counts: min={n_src_seen.min()}, "
              f"median={int(np.median(n_src_seen))}, max={n_src_seen.max()}")

        # -------------------------------
        # Train-set statistics for the cube and global targets
        # -------------------------------
        print("Computing train-set normalization stats...")
        train_items = held_targets["train"]
        if not train_items:
            raise RuntimeError("no training samples were built")

        cube_logs = []
        for cube, _, _ in train_items.values():
            cmask = cube > 0.0
            if np.any(cmask):
                cube_logs.append(np.log10(cube[cmask].astype(np.float64)))
        if not cube_logs:
            raise RuntimeError("no positive-valued cube voxels found in train set")
        cube_logs = np.concatenate(cube_logs)
        cube_mean = float(np.mean(cube_logs))
        cube_std = float(np.std(cube_logs) + 1e-6)

        mass_logs = np.array([m for _, m, _ in train_items.values()], dtype=np.float64)
        axes_all = np.stack([a for _, _, a in train_items.values()], axis=0).astype(np.float64)
        mass_mean = float(np.mean(mass_logs))
        mass_std = float(np.std(mass_logs) + 1e-6)
        axis_mean = np.mean(axes_all, axis=0)
        axis_std = np.std(axes_all, axis=0) + 1e-6

        print(f"3D density   log10 mean/std: {cube_mean:.6f}, {cube_std:.6f}")
        print(f"Cube mass log10(Msun) mean/std: {mass_mean:.6f}, {mass_std:.6f}")
        print(f"Axis lengths Mpc mean (a,b,c) : "
              f"{axis_mean[0]:.6f}, {axis_mean[1]:.6f}, {axis_mean[2]:.6f}")
        print(f"Axis lengths Mpc std  (a,b,c) : "
              f"{axis_std[0]:.6f}, {axis_std[1]:.6f}, {axis_std[2]:.6f}")

        for f in (f_train, f_val):
            f.attrs["cube_log10_mean"] = cube_mean
            f.attrs["cube_log10_std"] = cube_std
            f.attrs["cube_mass_log10_mean"] = mass_mean
            f.attrs["cube_mass_log10_std"] = mass_std
            f.attrs["axis_mean"] = axis_mean
            f.attrs["axis_std"] = axis_std

        # -------------------------------
        # Second pass: standardized cubes + global targets
        # -------------------------------
        for key in ("train", "val"):
            print(f"Writing {key} cubes/targets ({len(held_targets[key])} samples)...")
            for sid, (cube, mass_log10, axes) in held_targets[key].items():
                write_sample_targets(files[key], sid, cube, mass_log10, axes,
                                     cube_mean, cube_std, mass_mean, mass_std,
                                     axis_mean, axis_std)

    print("Done.")
    print(f"Train: {train_path} ({sid_counter['train']} samples)")
    print(f"Val  : {val_path} ({sid_counter['val']} samples)")


if __name__ == "__main__":
    main()
