#!/usr/bin/env python3
import os
import h5py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
import multiprocessing as mp

dirc_path = "/home/habjan.e/"
import sys
sys.path.append(dirc_path + "TNG/TNG_cluster_dynamics")
import TNG_DA


# ============================================================
# Configs
# ============================================================

@dataclass
class GridConfig:
    fov_mpc: float = 5.0              # cube/image span is [-fov_mpc, +fov_mpc]
    image_resolution: int = 16
    cube_resolution: int = 16
    vz_max_kms: float = 4000.0
    eps: float = 1e-12
    floor_value: float = -5.0         # value for empty / ultra-low voxels after normalization


@dataclass
class ScalingConfig:
    pos_mean: float = 0.0
    pos_std: float = 5.0
    vel_mean: float = 0.0
    vel_std: float = 800.0


@dataclass
class MassConfig:
    h: float = 0.7


# ============================================================
# Helpers
# ============================================================

def _bin2d_sum(
    x: np.ndarray, y: np.ndarray, w: np.ndarray,
    xlim: Tuple[float, float], ylim: Tuple[float, float],
    H: int, W: int
) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        y.astype(np.float64),
        x.astype(np.float64),
        bins=(H, W),
        range=(ylim, xlim),
        weights=w.astype(np.float64),
    )
    return hist.astype(np.float32)


def _bin2d_count(
    x: np.ndarray, y: np.ndarray,
    xlim: Tuple[float, float], ylim: Tuple[float, float],
    H: int, W: int
) -> np.ndarray:
    return _bin2d_sum(
        x, y, np.ones_like(x, dtype=np.float32),
        xlim, ylim, H, W
    )


def _bin3d_sum(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray,
    xlim: Tuple[float, float], ylim: Tuple[float, float], zlim: Tuple[float, float],
    Nx: int, Ny: int, Nz: int
) -> np.ndarray:
    """
    Returns array with shape (Nz, Ny, Nx), so indexing is [z, y, x].
    """
    hist, _ = np.histogramdd(
        sample=np.stack([z, y, x], axis=-1).astype(np.float64),
        bins=(Nz, Ny, Nx),
        range=(zlim, ylim, xlim),
        weights=w.astype(np.float64),
    )
    return hist.astype(np.float32)


def make_galaxy_map_xy(x: np.ndarray, y: np.ndarray, cfg: GridConfig) -> np.ndarray:
    H = W = cfg.image_resolution
    xlim = (-cfg.fov_mpc, cfg.fov_mpc)
    ylim = (-cfg.fov_mpc, cfg.fov_mpc)
    N = max(len(x), 1)
    w = np.full_like(x, 1.0 / N, dtype=np.float32)
    return _bin2d_sum(x, y, w, xlim, ylim, H, W)


def make_galaxy_vz_mean_xy(
    x: np.ndarray, y: np.ndarray, vz: np.ndarray, cfg: GridConfig
) -> np.ndarray:
    H = W = cfg.image_resolution
    xlim = (-cfg.fov_mpc, cfg.fov_mpc)
    ylim = (-cfg.fov_mpc, cfg.fov_mpc)

    vz_sum = _bin2d_sum(x, y, vz.astype(np.float32), xlim, ylim, H, W)
    count = _bin2d_count(x, y, xlim, ylim, H, W)

    mean_vz = np.zeros_like(vz_sum, dtype=np.float32)
    mask = count > 0
    mean_vz[mask] = vz_sum[mask] / count[mask]
    return mean_vz


def make_galaxy_vz_disp_xy(
    x: np.ndarray, y: np.ndarray, vz: np.ndarray, cfg: GridConfig
) -> np.ndarray:
    """
    Per-pixel LOS velocity dispersion map in the XY plane.

    Uses:
        sigma_vz^2 = <vz^2> - <vz>^2

    Empty pixels and 1-galaxy pixels are set to 0.
    """
    H = W = cfg.image_resolution
    xlim = (-cfg.fov_mpc, cfg.fov_mpc)
    ylim = (-cfg.fov_mpc, cfg.fov_mpc)

    vz = vz.astype(np.float32)
    vz_sum = _bin2d_sum(x, y, vz, xlim, ylim, H, W)
    vz2_sum = _bin2d_sum(x, y, vz**2, xlim, ylim, H, W)
    count = _bin2d_count(x, y, xlim, ylim, H, W)

    disp_vz = np.zeros((H, W), dtype=np.float32)

    # Require at least 2 galaxies in a pixel for a meaningful dispersion
    mask = count > 1
    if np.any(mask):
        mean = vz_sum[mask] / count[mask]
        mean2 = vz2_sum[mask] / count[mask]
        var = np.maximum(mean2 - mean**2, 0.0)
        disp_vz[mask] = np.sqrt(var).astype(np.float32)

    return disp_vz


def make_total_mass_map_xy(
    dm_x: np.ndarray, dm_y: np.ndarray, dm_m: np.ndarray,
    gas_x: np.ndarray, gas_y: np.ndarray, gas_m: np.ndarray,
    star_x: np.ndarray, star_y: np.ndarray, star_m: np.ndarray,
    bh_x: np.ndarray, bh_y: np.ndarray, bh_m: np.ndarray,
    cfg: GridConfig
) -> np.ndarray:
    H = W = cfg.image_resolution
    xlim = (-cfg.fov_mpc, cfg.fov_mpc)
    ylim = (-cfg.fov_mpc, cfg.fov_mpc)

    def inside_xy(x, y):
        return (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])

    total_mass = np.zeros((H, W), dtype=np.float32)

    for x, y, m in [
        (dm_x, dm_y, dm_m),
        (gas_x, gas_y, gas_m),
        (star_x, star_y, star_m),
        (bh_x, bh_y, bh_m),
    ]:
        if x.size == 0:
            continue
        keep = inside_xy(x, y)
        if np.any(keep):
            total_mass += _bin2d_sum(x[keep], y[keep], m[keep], xlim, ylim, H, W)

    pix_size = (xlim[1] - xlim[0]) / W
    pix_area = (pix_size ** 2) + cfg.eps
    sigma = total_mass / pix_area  # Msun / Mpc^2
    return sigma.astype(np.float32)


def make_total_density_cube(
    dm_pos: np.ndarray, dm_m: np.ndarray,
    gas_pos: np.ndarray, gas_m: np.ndarray,
    star_pos: np.ndarray, star_m: np.ndarray,
    bh_pos: np.ndarray, bh_m: np.ndarray,
    cfg: GridConfig
) -> np.ndarray:
    N = cfg.cube_resolution
    xlim = ylim = zlim = (-cfg.fov_mpc, cfg.fov_mpc)

    def inside_xyz(pos):
        return (
            (pos[:, 0] >= xlim[0]) & (pos[:, 0] <= xlim[1]) &
            (pos[:, 1] >= ylim[0]) & (pos[:, 1] <= ylim[1]) &
            (pos[:, 2] >= zlim[0]) & (pos[:, 2] <= zlim[1])
        )

    cube_mass = np.zeros((N, N, N), dtype=np.float32)  # (z, y, x)

    for pos, m in [
        (dm_pos, dm_m),
        (gas_pos, gas_m),
        (star_pos, star_m),
        (bh_pos, bh_m),
    ]:
        if pos.shape[0] == 0:
            continue
        keep = inside_xyz(pos)
        if np.any(keep):
            cube_mass += _bin3d_sum(
                pos[keep, 0], pos[keep, 1], pos[keep, 2], m[keep],
                xlim, ylim, zlim, N, N, N
            )

    voxel_size = (xlim[1] - xlim[0]) / N
    voxel_vol = (voxel_size ** 3) + cfg.eps
    rho = cube_mass / voxel_vol  # Msun / Mpc^3
    return rho.astype(np.float32)


def log_standardize_with_floor(
    x: np.ndarray, mean: float, std: float, floor_value: float, eps: float = 0.0
) -> np.ndarray:
    out = np.full_like(x, floor_value, dtype=np.float32)
    mask = x > eps
    if np.any(mask):
        v = np.log10(x[mask].astype(np.float64))
        z = (v - mean) / std
        z = np.maximum(z, floor_value)
        out[mask] = z.astype(np.float32)
    return out


# ============================================================
# Global cube targets: enclosed mass and shape-tensor axis lengths
# ============================================================

def rho_cube_to_mass_msun(rho_cube: np.ndarray, fov_mpc: float) -> float:
    """
    rho_cube: (Z,Y,X) density in Msun / Mpc^3
    Returns enclosed mass in Msun.
    """
    N = rho_cube.shape[0]
    voxel_size = (2.0 * fov_mpc) / N
    voxel_vol = voxel_size ** 3
    return float(np.sum(rho_cube.astype(np.float64)) * voxel_vol)


def rho_cube_to_axis_lengths_mpc(rho_cube: np.ndarray, fov_mpc: float) -> Tuple[float, float, float]:
    """
    rho_cube: (Z,Y,X) density in Msun / Mpc^3
    Returns mass-weighted shape-tensor axis lengths (a, b, c) in Mpc, sorted largest to smallest.
    """
    N = rho_cube.shape[0]
    voxel_size = (2.0 * fov_mpc) / N
    voxel_vol = voxel_size ** 3

    mass = rho_cube.astype(np.float64) * voxel_vol
    total_mass = np.sum(mass)

    if total_mass <= 0:
        return float("nan"), float("nan"), float("nan")

    coords_1d = (np.arange(N, dtype=np.float64) + 0.5) * voxel_size - fov_mpc
    z, y, x = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")

    x_com = np.sum(mass * x) / total_mass
    y_com = np.sum(mass * y) / total_mass
    z_com = np.sum(mass * z) / total_mass

    dx = x - x_com
    dy = y - y_com
    dz = z - z_com

    S_xx = np.sum(mass * dx * dx) / total_mass
    S_yy = np.sum(mass * dy * dy) / total_mass
    S_zz = np.sum(mass * dz * dz) / total_mass
    S_xy = np.sum(mass * dx * dy) / total_mass
    S_xz = np.sum(mass * dx * dz) / total_mass
    S_yz = np.sum(mass * dy * dz) / total_mass

    S = np.array([
        [S_xx, S_xy, S_xz],
        [S_xy, S_yy, S_yz],
        [S_xz, S_yz, S_zz],
    ], dtype=np.float64)

    evals = np.linalg.eigvalsh(S)
    evals = np.sort(evals)[::-1]
    a, b, c = np.sqrt(np.clip(evals, 0.0, None))
    return float(a), float(b), float(c)


def cube_to_mass_msun(cube_norm: np.ndarray, metadata: Dict) -> float:
    """
    cube_norm: (Z,Y,X) log-standardized cube as stored in the HDF5 file
    metadata:  dict of HDF5 file attrs (e.g. preload_hdf5_to_memory(...)["metadata"])

    Returns enclosed mass in Msun by inverting the cube normalization.
    """
    cube_mean = float(metadata["cube_log10_mean"])
    cube_std = float(metadata["cube_log10_std"])
    floor_value = float(metadata["floor_value"])
    fov_mpc = float(metadata["map_fov_mpc"])
    N = int(metadata["cube_resolution"])

    rho = np.zeros_like(cube_norm, dtype=np.float64)
    mask = cube_norm > floor_value + 1e-6
    rho[mask] = 10.0 ** (cube_norm[mask] * cube_std + cube_mean)

    voxel_size = (2.0 * fov_mpc) / N
    voxel_vol = voxel_size ** 3
    return float(np.sum(rho) * voxel_vol)


def cube_to_axis_lengths_mpc(cube_norm: np.ndarray, metadata: Dict) -> Tuple[float, float, float]:
    """
    cube_norm: (Z,Y,X) log-standardized cube as stored in the HDF5 file
    metadata:  dict of HDF5 file attrs (e.g. preload_hdf5_to_memory(...)["metadata"])

    Returns mass-weighted shape-tensor axis lengths (a, b, c) in Mpc, sorted largest to smallest,
    by inverting the cube normalization.
    """
    cube_mean = float(metadata["cube_log10_mean"])
    cube_std = float(metadata["cube_log10_std"])
    floor_value = float(metadata["floor_value"])
    fov_mpc = float(metadata["map_fov_mpc"])

    rho = np.zeros_like(cube_norm, dtype=np.float64)
    mask = cube_norm > floor_value + 1e-6
    rho[mask] = 10.0 ** (cube_norm[mask] * cube_std + cube_mean)

    return rho_cube_to_axis_lengths_mpc(rho, fov_mpc)


def xy_to_pixel_coords(x: np.ndarray, y: np.ndarray, cfg: GridConfig) -> np.ndarray:
    H = W = cfg.image_resolution
    xlim = (-cfg.fov_mpc, cfg.fov_mpc)
    ylim = (-cfg.fov_mpc, cfg.fov_mpc)

    x01 = (x - xlim[0]) / (xlim[1] - xlim[0] + cfg.eps)
    y01 = (y - ylim[0]) / (ylim[1] - ylim[0] + cfg.eps)

    x01 = np.clip(x01, 0.0, 1.0)
    y01 = np.clip(y01, 0.0, 1.0)

    x_pix = x01 * (W - 1)
    y_pix = y01 * (H - 1)

    return np.stack([x_pix, y_pix], axis=-1).astype(np.float32)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.uniform(-1.0, 1.0, size=(3,))
    n = np.linalg.norm(v)
    if n < 1e-8:
        return random_unit_vector(rng)
    return v / n


# ============================================================
# Sample builder
# ============================================================

def build_one_sample(args) -> Dict[str, np.ndarray]:
    (
        npz_path, sim, cluster_idx, proj_vec,
        grid_cfg, scaling_cfg, mass_cfg, max_nodes
    ) = args

    data = np.load(npz_path)

    # -------------------------------
    # Cluster-member galaxies
    # -------------------------------
    bright_bool = data["sub_massTotal"][:, 4] != 0

    boxsize = 400.0
    eps = 1e-9

    difpos = np.subtract(data["sub_pos"][bright_bool], data["CoP"])
    coordinates = (difpos + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    pos = (coordinates / (data["h"] * data["a"])) + eps  # Mpc

    vel = data["sub_vel"][bright_bool, :]
    v_bulk = np.mean(vel, axis=0)
    vel = vel - v_bulk

    ro_pos, ro_vel = TNG_DA.rotate_to_viewing_frame(pos, vel, proj_vec)

    x = ro_pos[:, 0].astype(np.float32)
    y = ro_pos[:, 1].astype(np.float32)
    z = ro_pos[:, 2].astype(np.float32)
    vx = ro_vel[:, 0].astype(np.float32)
    vy = ro_vel[:, 1].astype(np.float32)
    vz = ro_vel[:, 2].astype(np.float32)

    # -------------------------------
    # Matter components: positions + masses
    # Assumed masses stored in Msun / h
    # -------------------------------
    def prep_component(pos_key, mass_key):
        if pos_key not in data or mass_key not in data:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        pos_raw = data[pos_key].astype(np.float64)
        mass_raw = data[mass_key].astype(np.float64)

        if pos_raw.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        dif = np.subtract(pos_raw, data["CoP"])
        coords = (dif + 0.5 * boxsize) % boxsize - 0.5 * boxsize
        pos_mpc = (coords / (data["h"] * data["a"])) + eps

        # Convert Msun / h -> Msun
        mass_msun = mass_raw / mass_cfg.h

        rot_pos, _ = TNG_DA.rotate_to_viewing_frame(pos_mpc, np.zeros_like(pos_mpc), proj_vec)
        return rot_pos.astype(np.float32), mass_msun.astype(np.float32)

    dm_pos, dm_m = prep_component("dm_pos", "dm_mass")
    gas_pos, gas_m = prep_component("gas_pos", "gas_mass")
    star_pos, star_m = prep_component("star_pos", "star_mass")
    bh_pos, bh_m = prep_component("bh_pos", "bh_mass")

    # -------------------------------
    # 2D conditioning images
    # -------------------------------
    mass_xy = make_total_mass_map_xy(
        dm_pos[:, 0], dm_pos[:, 1], dm_m,
        gas_pos[:, 0], gas_pos[:, 1], gas_m,
        star_pos[:, 0], star_pos[:, 1], star_m,
        bh_pos[:, 0], bh_pos[:, 1], bh_m,
        grid_cfg
    )

    gal_xy = make_galaxy_map_xy(x, y, grid_cfg)

    gal_vz_xy = make_galaxy_vz_mean_xy(x, y, vz, grid_cfg)
    gal_vz_xy = (gal_vz_xy - scaling_cfg.vel_mean) / scaling_cfg.vel_std

    gal_vz_disp_xy = make_galaxy_vz_disp_xy(x, y, vz, grid_cfg)
    gal_vz_disp_xy = gal_vz_disp_xy / scaling_cfg.vel_std

    # -------------------------------
    # 3D target cube
    # -------------------------------
    density_cube = make_total_density_cube(
        dm_pos, dm_m,
        gas_pos, gas_m,
        star_pos, star_m,
        bh_pos, bh_m,
        grid_cfg
    )

    # -------------------------------
    # Global cube targets: log10 enclosed mass and shape-tensor axis lengths.
    # Computed from the raw density cube (Msun/Mpc^3) to avoid floor roundtrip error.
    # -------------------------------
    cube_mass_msun = rho_cube_to_mass_msun(density_cube, grid_cfg.fov_mpc)
    cube_mass_log10_msun = np.log10(max(cube_mass_msun, 1e-30))
    a_mpc, b_mpc, c_mpc = rho_cube_to_axis_lengths_mpc(density_cube, grid_cfg.fov_mpc)
    axis_lengths_mpc = np.array([a_mpc, b_mpc, c_mpc], dtype=np.float32)

    # -------------------------------
    # Galaxy catalog save
    # -------------------------------
    gal_pixel_coords = xy_to_pixel_coords(x, y, grid_cfg)

    x_s = (x - scaling_cfg.pos_mean) / scaling_cfg.pos_std
    y_s = (y - scaling_cfg.pos_mean) / scaling_cfg.pos_std
    z_s = (z - scaling_cfg.pos_mean) / scaling_cfg.pos_std

    vx_s = (vx - scaling_cfg.vel_mean) / scaling_cfg.vel_std
    vy_s = (vy - scaling_cfg.vel_mean) / scaling_cfg.vel_std
    vz_s = (vz - scaling_cfg.vel_mean) / scaling_cfg.vel_std

    N_gal = x.shape[0]
    n_feat = np.full((N_gal,), float(N_gal), dtype=np.float32)

    # observed + hidden catalog
    gal_features = np.stack([x_s, y_s, vz_s, n_feat], axis=-1).astype(np.float32)
    gal_targets = np.stack([z_s, vx_s, vy_s], axis=-1).astype(np.float32)

    N = min(N_gal, max_nodes)
    feat_pad = np.zeros((max_nodes, 4), dtype=np.float32)
    targ_pad = np.zeros((max_nodes, 3), dtype=np.float32)
    pix_pad = np.zeros((max_nodes, 2), dtype=np.float32)
    mask = np.zeros((max_nodes,), dtype=np.float32)

    feat_pad[:N] = gal_features[:N]
    targ_pad[:N] = gal_targets[:N]
    pix_pad[:N] = gal_pixel_coords[:N]
    mask[:N] = 1.0

    return dict(
        raw_mass_xy=mass_xy,
        raw_density_cube=density_cube,
        gal_xy=gal_xy.astype(np.float32),
        gal_vz_xy=gal_vz_xy.astype(np.float32),
        gal_vz_disp_xy=gal_vz_disp_xy.astype(np.float32),

        gal_features=feat_pad,
        gal_targets=targ_pad,
        gal_pixel_coords=pix_pad,
        mask=mask,

        cube_mass_log10_msun=np.float32(cube_mass_log10_msun),
        axis_lengths_mpc=axis_lengths_mpc,

        n_gal=np.int32(N_gal),
        proj_vec=np.asarray(proj_vec, dtype=np.float32),
        sim=str(sim),
        cluster_idx=np.int32(cluster_idx),
        halo_mass=np.float32(data["Mfof"]),
    )


# ============================================================
# Writing
# ============================================================

def write_sample_to_hdf5(
    h5: h5py.File,
    sample_id: int,
    sample: Dict[str, np.ndarray],
    mass_mean: float,
    mass_std: float,
    cube_mean: float,
    cube_std: float,
    floor_value: float,
    cube_mass_log10_mean: float,
    cube_mass_log10_std: float,
    axis_mean: np.ndarray,
    axis_std: np.ndarray,
):
    grp = h5.create_group(f"{sample_id:06d}")
    grp.attrs["id"] = int(sample_id)
    grp.attrs["simulation"] = sample["sim"]
    grp.attrs["cluster_index"] = int(sample["cluster_idx"])
    grp.attrs["cluster_mass"] = float(sample["halo_mass"])
    grp.attrs["n_galaxies"] = int(sample["n_gal"])

    mass_xy = log_standardize_with_floor(
        sample["raw_mass_xy"], mass_mean, mass_std, floor_value=floor_value
    )
    cube = log_standardize_with_floor(
        sample["raw_density_cube"], cube_mean, cube_std, floor_value=floor_value
    )

    images = np.stack(
        [
            mass_xy,
            sample["gal_xy"],
            sample["gal_vz_xy"],
            sample["gal_vz_disp_xy"],
        ],
        axis=0
    ).astype(np.float32)

    raw_mass_log10 = float(sample["cube_mass_log10_msun"])
    raw_axes = np.asarray(sample["axis_lengths_mpc"], dtype=np.float32)

    std_mass = (raw_mass_log10 - cube_mass_log10_mean) / cube_mass_log10_std
    std_axes = (raw_axes.astype(np.float64) - axis_mean) / axis_std
    globals_target = np.concatenate(
        [np.array([std_mass], dtype=np.float32), std_axes.astype(np.float32)],
        axis=0,
    )

    grp.create_dataset("projection_vector", data=sample["proj_vec"])
    grp.create_dataset("images", data=images, compression="gzip")               # (4,H,W)
    grp.create_dataset("density_cube", data=cube, compression="gzip")           # (Z,Y,X)
    grp.create_dataset("gal_features", data=sample["gal_features"], compression="gzip")
    grp.create_dataset("gal_targets", data=sample["gal_targets"], compression="gzip")
    grp.create_dataset("gal_pixel_coords", data=sample["gal_pixel_coords"], compression="gzip")
    grp.create_dataset("mask", data=sample["mask"], compression="gzip")

    grp.create_dataset("cube_mass_log10_msun", data=np.float32(raw_mass_log10))
    grp.create_dataset("axis_lengths_mpc", data=raw_axes)
    grp.create_dataset("globals_target", data=globals_target)


# ============================================================
# Main
# ============================================================

def main():
    data_root = "/projects/mccleary_group/habjan.e/TNG/Data/"
    out_dir = "/projects/mccleary_group/habjan.e/TNG/Data/conditional_diffusion_data/"
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "cond_diffusion_16cubed_16img_train.h5")
    test_path = os.path.join(out_dir, "cond_diffusion_16cubed_16img_test.h5")

    for p in [train_path, test_path]:
        if os.path.exists(p):
            os.remove(p)

    cluster_inds = np.array([f"{i:03d}" for i in range(1, 101)])
    simulations = np.array(["SIDM0.1b", "SIDM0.3b", "vdSIDMb", "CDMb"])

    dataset_size = 10**4
    num_clusters = cluster_inds.shape[0] * simulations.shape[0]
    num_proj_per_cluster = int(dataset_size / num_clusters)

    rng = np.random.default_rng(42)
    test_size = int(len(cluster_inds) * 0.1)
    subset_test = rng.choice(cluster_inds, size=test_size, replace=False)
    subset_test_set = set(subset_test.tolist())

    max_nodes = 700
    grid_cfg = GridConfig(
        fov_mpc=5.0,
        image_resolution=16,
        cube_resolution=16,
        vz_max_kms=4000.0,
        floor_value=-5.0,
    )
    scaling_cfg = ScalingConfig(
        pos_mean=0.0,
        pos_std=5.0,
        vel_mean=0.0,
        vel_std=800.0,
    )
    mass_cfg = MassConfig(h=0.7)

    jobs = []
    for cluster_idx in cluster_inds:
        for sim in simulations:
            npz_path = os.path.join(data_root, sim, f"GrNm_{cluster_idx}.npz")
            for _ in range(num_proj_per_cluster):
                proj_vec = random_unit_vector(rng)
                jobs.append(
                    (npz_path, sim, int(cluster_idx), proj_vec, grid_cfg, scaling_cfg, mass_cfg, max_nodes)
                )

    print(f"Building {len(jobs)} raw samples...")

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    raw_train_samples: List[Dict[str, np.ndarray]] = []
    raw_test_samples: List[Dict[str, np.ndarray]] = []

    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        for i, sample in enumerate(pool.imap_unordered(build_one_sample, jobs, chunksize=4), start=1):
            cluster_idx_str = f"{int(sample['cluster_idx']):03d}"
            is_test = cluster_idx_str in subset_test_set

            if is_test:
                raw_test_samples.append(sample)
            else:
                raw_train_samples.append(sample)

            if i % 250 == 0:
                print(f"  built {i}/{len(jobs)} samples...")

    print("Computing global train-set normalization stats...")

    train_mass_logs = []
    train_cube_logs = []

    for s in raw_train_samples:
        m = s["raw_mass_xy"]
        c = s["raw_density_cube"]

        mmask = m > 0.0
        cmask = c > 0.0

        if np.any(mmask):
            train_mass_logs.append(np.log10(m[mmask].astype(np.float64)))
        if np.any(cmask):
            train_cube_logs.append(np.log10(c[cmask].astype(np.float64)))

    if len(train_mass_logs) == 0 or len(train_cube_logs) == 0:
        raise RuntimeError("Could not compute normalization stats: no positive-valued mass pixels/voxels found.")

    train_mass_logs = np.concatenate(train_mass_logs)
    train_cube_logs = np.concatenate(train_cube_logs)

    mass_mean = float(np.mean(train_mass_logs))
    mass_std = float(np.std(train_mass_logs) + 1e-6)
    cube_mean = float(np.mean(train_cube_logs))
    cube_std = float(np.std(train_cube_logs) + 1e-6)

    print(f"Projected mass log10 mean/std: {mass_mean:.6f}, {mass_std:.6f}")
    print(f"3D density   log10 mean/std: {cube_mean:.6f}, {cube_std:.6f}")

    # Global cube targets (mass + 3 axis lengths) stats over train set.
    train_mass_log10_globals = np.array(
        [float(s["cube_mass_log10_msun"]) for s in raw_train_samples],
        dtype=np.float64,
    )
    train_axes = np.stack(
        [np.asarray(s["axis_lengths_mpc"], dtype=np.float64) for s in raw_train_samples],
        axis=0,
    )  # (Ntrain, 3)

    cube_mass_log10_mean = float(np.mean(train_mass_log10_globals))
    cube_mass_log10_std = float(np.std(train_mass_log10_globals) + 1e-6)

    axis_mean = np.mean(train_axes, axis=0).astype(np.float64)         # (3,)
    axis_std = (np.std(train_axes, axis=0) + 1e-6).astype(np.float64)  # (3,)

    print(f"Cube mass log10(Msun) mean/std: {cube_mass_log10_mean:.6f}, {cube_mass_log10_std:.6f}")
    print(f"Axis lengths Mpc mean (a,b,c) : {axis_mean[0]:.6f}, {axis_mean[1]:.6f}, {axis_mean[2]:.6f}")
    print(f"Axis lengths Mpc std  (a,b,c) : {axis_std[0]:.6f}, {axis_std[1]:.6f}, {axis_std[2]:.6f}")

    with h5py.File(train_path, "w") as f_train, h5py.File(test_path, "w") as f_test:
        for f in [f_train, f_test]:
            f.attrs["map_fov_mpc"] = grid_cfg.fov_mpc
            f.attrs["image_resolution"] = grid_cfg.image_resolution
            f.attrs["cube_resolution"] = grid_cfg.cube_resolution
            f.attrs["vz_max_kms"] = grid_cfg.vz_max_kms
            f.attrs["floor_value"] = grid_cfg.floor_value
            f.attrs["max_nodes"] = max_nodes

            f.attrs["pos_mean"] = scaling_cfg.pos_mean
            f.attrs["pos_std"] = scaling_cfg.pos_std
            f.attrs["vel_mean"] = scaling_cfg.vel_mean
            f.attrs["vel_std"] = scaling_cfg.vel_std

            f.attrs["h"] = mass_cfg.h
            f.attrs["proj_mass_log10_mean"] = mass_mean
            f.attrs["proj_mass_log10_std"] = mass_std
            f.attrs["cube_log10_mean"] = cube_mean
            f.attrs["cube_log10_std"] = cube_std

            f.attrs["cube_mass_log10_mean"] = cube_mass_log10_mean
            f.attrs["cube_mass_log10_std"] = cube_mass_log10_std
            f.attrs["axis_mean"] = axis_mean.astype(np.float64)
            f.attrs["axis_std"] = axis_std.astype(np.float64)
            f.attrs["globals_target_columns"] = np.array(
                [b"mass_log10_msun", b"axis_a_mpc", b"axis_b_mpc", b"axis_c_mpc"],
                dtype="S20",
            )

            f.attrs["image_channels"] = np.array(
                [b"mass_xy", b"gal_xy", b"gal_vz_xy", b"gal_vz_disp_xy"],
                dtype="S20"
            )
            f.attrs["density_cube_order"] = "zyx"
            f.attrs["gal_feature_columns"] = np.array([b"x", b"y", b"vz", b"Ngal"], dtype="S8")
            f.attrs["gal_target_columns"] = np.array([b"z", b"vx", b"vy"], dtype="S8")
            f.attrs["gal_pixel_coords_columns"] = np.array([b"x_pix", b"y_pix"], dtype="S8")

        print("Writing train file...")
        for sid, sample in enumerate(raw_train_samples):
            write_sample_to_hdf5(
                f_train, sid, sample,
                mass_mean, mass_std, cube_mean, cube_std,
                grid_cfg.floor_value,
                cube_mass_log10_mean, cube_mass_log10_std,
                axis_mean, axis_std,
            )
            if (sid + 1) % 250 == 0:
                print(f"  wrote train {sid + 1}/{len(raw_train_samples)}")

        print("Writing test file...")
        for sid, sample in enumerate(raw_test_samples):
            write_sample_to_hdf5(
                f_test, sid, sample,
                mass_mean, mass_std, cube_mean, cube_std,
                grid_cfg.floor_value,
                cube_mass_log10_mean, cube_mass_log10_std,
                axis_mean, axis_std,
            )
            if (sid + 1) % 250 == 0:
                print(f"  wrote test {sid + 1}/{len(raw_test_samples)}")

    print("Done.")
    print(f"Train: {train_path}")
    print(f"Test : {test_path}")


if __name__ == "__main__":
    main()