import os
import h5py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import multiprocessing as mp

dirc_path = "/home/habjan.e/"
import sys
sys.path.append(dirc_path + "TNG/TNG_cluster_dynamics")
import TNG_DA


@dataclass
class MapConfig:
    fov_mpc: float = 6.5
    vz_max_kms: float = 4000.0
    resolution: int = 128
    eps: float = 1e-12

@dataclass
class ScalingConfig:
    ### global scaling for inputs/targets
    pos_mean: float = 0.0
    pos_std: float = 1.5
    pos_mag_mean: float = 1.25
    pos_mag_std: float = 1.25
    vel_mean: float = 0.0
    vel_std: float = 800.0
    sigma_mean: float = 13.0
    sigma_std: float = 0.5

@dataclass
class MassConfig:
    m_dm_msun_over_h: float = 5.5e9
    h: float = 0.7

    @property
    def m_dm_msun(self) -> float:
        return self.m_dm_msun_over_h / self.h


def _bin2d_sum(x: np.ndarray, y: np.ndarray, w: np.ndarray,
              xlim: Tuple[float, float], ylim: Tuple[float, float],
              H: int, W: int) -> np.ndarray:
    """
    Sum weights w into a 2D grid using numpy.histogram2d.
    Returns array shape (H, W) with y as first axis, x as second axis.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    w = w.astype(np.float64)

    hist, yedges, xedges = np.histogram2d(
        y, x,
        bins=(H, W),
        range=(ylim, xlim),
        weights=w
    )
    return hist.astype(np.float32)

def _bin2d_count(x: np.ndarray, y: np.ndarray,
                xlim: Tuple[float, float], ylim: Tuple[float, float],
                H: int, W: int) -> np.ndarray:
    hist = _bin2d_sum(x, y, np.ones_like(x, dtype=np.float32), xlim, ylim, H, W)
    return hist

def make_galaxy_map_xy(x: np.ndarray, y: np.ndarray, cfg: MapConfig) -> np.ndarray:
    """Galaxy occupancy in xy with 1/N per galaxy per occupied pixel (sums to ~1 if no collisions)."""
    H = W = cfg.resolution
    xlim = (-cfg.fov_mpc, cfg.fov_mpc)
    ylim = (-cfg.fov_mpc, cfg.fov_mpc)
    N = max(len(x), 1)
    w = np.full_like(x, 1.0 / N, dtype=np.float32)
    return _bin2d_sum(x, y, w, xlim, ylim, H, W)

def make_galaxy_vz_mean_xy(x: np.ndarray, y: np.ndarray, vz: np.ndarray,
                           cfg: MapConfig) -> np.ndarray:
    """
    Mean vz per xy pixel.
    """
    H = W = cfg.resolution
    xlim = (-cfg.fov_mpc, cfg.fov_mpc)
    ylim = (-cfg.fov_mpc, cfg.fov_mpc)

    vz_sum = _bin2d_sum(x, y, vz.astype(np.float32), xlim, ylim, H, W)
    count = _bin2d_count(x, y, xlim, ylim, H, W)

    mean_vz = np.zeros_like(vz_sum, dtype=np.float32)
    mask = count > 0
    mean_vz[mask] = vz_sum[mask] / count[mask]

    return mean_vz

def make_galaxy_map_xvz(x: np.ndarray, vz: np.ndarray, cfg: MapConfig) -> np.ndarray:
    """Galaxy occupancy in x-vz with 1/N per galaxy (image axes: vz vertical, x horizontal)."""
    H = W = cfg.resolution
    xlim = (-cfg.fov_mpc, cfg.fov_mpc)
    vlim = (-cfg.vz_max_kms, cfg.vz_max_kms)
    N = max(len(x), 1)
    w = np.full_like(x, 1.0 / N, dtype=np.float32)
    return _bin2d_sum(x, vz, w, xlim, vlim, H, W)

def make_galaxy_map_yvz(y: np.ndarray, vz: np.ndarray, cfg: MapConfig) -> np.ndarray:
    """Galaxy occupancy in y-vz with 1/N per galaxy (image axes: vz vertical, y horizontal)."""
    H = W = cfg.resolution
    ylim = (-cfg.fov_mpc, cfg.fov_mpc)
    vlim = (-cfg.vz_max_kms, cfg.vz_max_kms)
    N = max(len(y), 1)
    w = np.full_like(y, 1.0 / N, dtype=np.float32)
    return _bin2d_sum(y, vz, w, ylim, vlim, H, W)

def make_mass_map_xy(dm_x: np.ndarray, dm_y: np.ndarray, cfg: MapConfig, mass_cfg: MassConfig,
                     scaling: ScalingConfig) -> np.ndarray:
    """
    Projected surface density Sigma(x,y) in Msun / Mpc^2 on the xy plane.
    - Bins dm particles into pixels, sums mass, then divides by pixel area.
    - Optional z-score scaling if scaling.sigma_mean/std are set.
    """
    H = W = cfg.resolution
    xlim = (-cfg.fov_mpc, cfg.fov_mpc)
    ylim = (-cfg.fov_mpc, cfg.fov_mpc)

    m = mass_cfg.m_dm_msun
    inside = (dm_x >= xlim[0]) & (dm_x <= xlim[1]) & (dm_y >= ylim[0]) & (dm_y <= ylim[1])
    dm_x = dm_x[inside]
    dm_y = dm_y[inside]

    if dm_x.size == 0:
        sigma = np.zeros((H, W), dtype=np.float32)
    else:
        mass_per_pix = _bin2d_sum(dm_x, dm_y, np.full_like(dm_x, m, dtype=np.float32), xlim, ylim, H, W)
        pix_size = (xlim[1] - xlim[0]) / W  # Mpc
        pix_area = (pix_size ** 2) + cfg.eps
        sigma = (mass_per_pix / pix_area).astype(np.float32)  # Msun / Mpc^2

    return sigma

def zscore_log_mass_with_empty(mass_map, scaling_cfg, empty_value=-5.0, eps=0.0, clip=5.0):
    """
    mass_map: (H,W) projected mass per pixel (>=0)
    Returns:
      zmap: (H,W) float32, with zeros set to empty_value
      mu, sig: stats of log10(nonzero masses)
    """
    mask = mass_map > eps
    zmap = np.full_like(mass_map, fill_value=empty_value, dtype=np.float32)

    if np.any(mask):
        v = np.log10(mass_map[mask].astype(np.float64))
        z = (v - scaling_cfg.sigma_mean) / scaling_cfg.sigma_std
        if clip is not None:
            z = np.clip(z, -clip, clip)
        zmap[mask] = z.astype(np.float32)
        return zmap

    return zmap


def xy_to_pixel_coords(x: np.ndarray, y: np.ndarray, cfg: MapConfig) -> np.ndarray:
    """
    Convert physical (x,y) [Mpc] into continuous image pixel coordinates for sampler queries.

    Returns:
        pix_coords: (N, 2) float32
            [:,0] = x_pix in [0, W-1]
            [:,1] = y_pix in [0, H-1]

    These are continuous coords (not integer indices), so they work for bilinear sampling.
    Convention matches the same x/y limits used to build the xy maps.
    """
    H = W = cfg.resolution
    xlim = (-cfg.fov_mpc, cfg.fov_mpc)
    ylim = (-cfg.fov_mpc, cfg.fov_mpc)

    x01 = (x - xlim[0]) / (xlim[1] - xlim[0] + cfg.eps)
    y01 = (y - ylim[0]) / (ylim[1] - ylim[0] + cfg.eps)

    x01 = np.clip(x01, 0.0, 1.0)
    y01 = np.clip(y01, 0.0, 1.0)

    x_pix = x01 * (W - 1)
    y_pix = y01 * (H - 1)

    return np.stack([x_pix, y_pix], axis=-1).astype(np.float32)


def build_one_sample(args) -> Dict[str, np.ndarray]:
    """
    Worker function.
    Returns a dict with padded arrays and metadata; writing is done in main.
    """
    (npz_path, sim, cluster_idx, proj_vec, map_cfg, scaling_cfg, mass_cfg, max_nodes) = args

    data = np.load(npz_path)

    ### Remove DM-only subhalos
    bright_bool = data["sub_massTotal"][:, 4] != 0

    ### PBCs
    boxsize = 400
    difpos = np.subtract(data['sub_pos'][bright_bool], data['CoP'])
    coordinates= (difpos + 0.5 * boxsize) % boxsize - 0.5 * boxsize

    eps = 1e-9
    # c Mpc / h
    pos = (coordinates / (data["h"] * data["a"])) + eps
    # km/s
    vel = data["sub_vel"][bright_bool, :]
    v_bulk = np.mean(vel, axis=0)
    vel = vel - v_bulk

    # DM particles
    dm_pos = data["dm_pos"].astype(np.float64)
    difdm = np.subtract(dm_pos, data["CoP"])
    dm_coordinates = (difdm + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    dm_pos_mpc = (dm_coordinates / (data["h"] * data["a"])) + eps  # Mpc

    # rotate to viewing frame
    ro_pos, ro_vel = TNG_DA.rotate_to_viewing_frame(pos, vel, proj_vec)
    ro_dm_pos, _ = TNG_DA.rotate_to_viewing_frame(dm_pos_mpc, np.zeros_like(dm_pos_mpc), proj_vec)

    # Use physical rotated coords for image making
    x = ro_pos[:, 0].astype(np.float32)
    y = ro_pos[:, 1].astype(np.float32)
    z = ro_pos[:, 2].astype(np.float32)
    vx = ro_vel[:, 0].astype(np.float32)
    vy = ro_vel[:, 1].astype(np.float32)
    vz = ro_vel[:, 2].astype(np.float32)

    # Build images
    mass_xy = make_mass_map_xy(ro_dm_pos[:, 0].astype(np.float32), ro_dm_pos[:, 1].astype(np.float32),
                               map_cfg, mass_cfg, scaling_cfg)
    mass_xy = zscore_log_mass_with_empty(mass_xy, scaling_cfg = scaling_cfg, empty_value=-5.0, eps=0.0, clip=5.0)

    gal_xy = make_galaxy_map_xy(x, y, map_cfg)
    #gal_xvz = make_galaxy_map_xvz(x, vz, map_cfg)
    #gal_yvz = make_galaxy_map_yvz(y, vz, map_cfg)
    gal_vz_xy = make_galaxy_vz_mean_xy(x, y, vz, map_cfg)

    gal_vz_xy = (gal_vz_xy - scaling_cfg.vel_mean) / scaling_cfg.vel_std

    #images = np.stack([mass_xy, gal_xy, gal_xvz, gal_yvz], axis=0).astype(np.float32)  # (4,H,W)
    images = np.stack([mass_xy, gal_xy, gal_vz_xy], axis=0).astype(np.float32)  # (3,H,W)

    gal_pixel_coords = xy_to_pixel_coords(x, y, map_cfg)

    # Global scaling
    x_s = (x - scaling_cfg.pos_mean) / scaling_cfg.pos_std
    y_s = (y - scaling_cfg.pos_mean) / scaling_cfg.pos_std
    z_s = (z - scaling_cfg.pos_mean) / scaling_cfg.pos_std

    vx_s = (vx - scaling_cfg.vel_mean) / scaling_cfg.vel_std
    vy_s = (vy - scaling_cfg.vel_mean) / scaling_cfg.vel_std
    vz_s = (vz - scaling_cfg.vel_mean) / scaling_cfg.vel_std

    z_mag = (abs(z) - scaling_cfg.pos_mag_mean) / scaling_cfg.pos_mag_std

    N_gal = x.shape[0]
    n_feat = np.full((N_gal,), float(N_gal), dtype=np.float32)

    # (x,y,vz,Ngal)
    gal_features = np.stack([x_s, y_s, vz_s, n_feat], axis=-1).astype(np.float32)

    #targets = np.stack([z_s, vx_s, vy_s], axis=-1).astype(np.float32)
    targets = np.expand_dims(z_mag, axis=-1).astype(np.float32)

    # Pad
    N = min(N_gal, max_nodes)
    feat_pad = np.zeros((max_nodes, 4), dtype=np.float32)
    pix_pad  = np.zeros((max_nodes, 2), dtype=np.float32)
    #targ_pad = np.zeros((max_nodes, 3), dtype=np.float32)
    targ_pad = np.zeros((max_nodes, 1), dtype=np.float32)
    mask = np.zeros((max_nodes,), dtype=np.float32)

    feat_pad[:N] = gal_features[:N]
    pix_pad[:N]  = gal_pixel_coords[:N]
    targ_pad[:N] = targets[:N]
    mask[:N] = 1.0

    return dict(
        images=images,
        gal_features=feat_pad,
        gal_pixel_coords=pix_pad,
        targets=targ_pad,
        mask=mask,
        n_gal=np.int32(N_gal),
        proj_vec=np.asarray(proj_vec, dtype=np.float32),
        sim=str(sim),
        cluster_idx=np.int32(cluster_idx),
        halo_mass=np.float32(data["Mfof"]),
    )


def write_sample_to_hdf5(h5: h5py.File, sample_id: int, sample: Dict[str, np.ndarray]):
    grp = h5.create_group(f"{sample_id:06d}")
    grp.attrs["id"] = int(sample_id)
    grp.attrs["simulation"] = sample["sim"]
    grp.attrs["cluster_index"] = int(sample["cluster_idx"])
    grp.attrs["cluster_mass"] = float(sample["halo_mass"])
    grp.attrs["n_galaxies"] = int(sample["n_gal"])

    grp.create_dataset("projection_vector", data=sample["proj_vec"])
    grp.create_dataset("images", data=sample["images"], compression="gzip")
    grp.create_dataset("gal_features", data=sample["gal_features"], compression="gzip")
    grp.create_dataset("gal_pixel_coords", data=sample["gal_pixel_coords"], compression="gzip")
    grp.create_dataset("targets", data=sample["targets"], compression="gzip")
    grp.create_dataset("mask", data=sample["mask"], compression="gzip")


def main():
    # Paths
    data_root = "/projects/mccleary_group/habjan.e/TNG/Data/"
    out_dir = "/projects/mccleary_group/habjan.e/TNG/Data/CNN_MLP_data/"
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "CNN_MLP_train.h5")
    test_path = os.path.join(out_dir, "CNN_MLP_test.h5")

    # Overwrite if exists
    for p in [train_path, test_path]:
        if os.path.exists(p):
            os.remove(p)

    # Dataset definition (same as yours)
    cluster_inds = np.array([f"{i:03d}" for i in range(1, 101)])
    simulations = np.array(["SIDM0.1b", "SIDM0.3b", "vdSIDMb", "CDMb"])

    dataset_size = 10**4
    num_clusters = cluster_inds.shape[0] * simulations.shape[0]
    num_proj_per_cluster = int(dataset_size / num_clusters)

    # Train/test split by cluster id
    test_size = int(len(cluster_inds) * 0.1)
    subset_test = np.random.choice(cluster_inds, size=test_size, replace=False)
    subset_test_set = set(subset_test.tolist())

    # Config
    MAX_NODES = 700
    map_cfg = MapConfig(fov_mpc=3.5, vz_max_kms=4000.0, resolution=128)
    scaling_cfg = ScalingConfig(
        pos_mean=0.0, pos_std=1.5,
        pos_mag_mean=1.25, pos_mag_std=1.25,
        vel_mean=0.0, vel_std=800.0,
        sigma_mean=13.0, sigma_std=0.5,
    )
    mass_cfg = MassConfig(m_dm_msun_over_h=5.5e9, h=0.7)

    # Build job list
    jobs = []
    for cluster_idx in cluster_inds:
        for sim in simulations:
            npz_path = os.path.join(data_root, sim, f"GrNm_{cluster_idx}.npz")
            for _ in range(num_proj_per_cluster):
                proj_vec = np.random.uniform(-1, 1, (3,))
                jobs.append((npz_path, sim, int(cluster_idx), proj_vec, map_cfg, scaling_cfg, mass_cfg, MAX_NODES))

    # Parallel build
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"Building {len(jobs)} samples with {n_workers} workers...")
    sample_id = 0

    with h5py.File(train_path, "a") as f_train, h5py.File(test_path, "a") as f_test:
        # Store config metadata
        for f in [f_train, f_test]:
            f.attrs["map_fov_mpc"] = map_cfg.fov_mpc
            f.attrs["map_vz_max_kms"] = map_cfg.vz_max_kms
            f.attrs["map_resolution"] = map_cfg.resolution
            f.attrs["max_nodes"] = MAX_NODES

            f.attrs["pos_mean"] = scaling_cfg.pos_mean
            f.attrs["pos_std"] = scaling_cfg.pos_std
            f.attrs["vel_mean"] = scaling_cfg.vel_mean
            f.attrs["vel_std"] = scaling_cfg.vel_std

            f.attrs["dm_particle_mass_msun_over_h"] = mass_cfg.m_dm_msun_over_h
            f.attrs["h"] = mass_cfg.h
            f.attrs["dm_particle_mass_msun"] = mass_cfg.m_dm_msun

            f.attrs["sigma_mean"] = scaling_cfg.sigma_mean
            f.attrs["sigma_std"] = scaling_cfg.sigma_std

            # NEW: metadata for sampler coordinates
            f.attrs["gal_pixel_coords_columns"] = np.array([b"x_pix", b"y_pix"], dtype="S8")
            f.attrs["gal_pixel_coords_note"] = "Continuous pixel coords in xy image frame, range [0, resolution-1]"

        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            for sample in pool.imap_unordered(build_one_sample, jobs, chunksize=4):
                # Decide train/test by cluster id
                cluster_idx_str = f"{int(sample['cluster_idx']):03d}"
                is_test = cluster_idx_str in subset_test_set

                if is_test:
                    write_sample_to_hdf5(f_test, sample_id, sample)
                else:
                    write_sample_to_hdf5(f_train, sample_id, sample)

                sample_id += 1
                if sample_id % 500 == 0:
                    print(f"  wrote {sample_id}/{len(jobs)} samples...")

    print("Done.")
    print(f"Train: {train_path}")
    print(f"Test : {test_path}")


if __name__ == "__main__":
    main()