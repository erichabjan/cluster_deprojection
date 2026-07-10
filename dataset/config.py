"""
Shared configuration for the shape_dynamics dataset pipeline.

Pipeline (each stage has its own slurm script):
  Prep (run once)      : build_lovoccs_redshift_template.py  (make_lovoccs_template.slurm)
  Dataset build        : build_shape_dynamics_dataset.py     (build_dataset.slurm)

Merges the pieces of jaxlense_dataset/config.py and
conditional_diffusion_model/conditional_diffusion_data.py that this
pipeline actually uses.
"""
from dataclasses import dataclass
from typing import Tuple


# Empirical source-redshift templates, selectable via
# LensingConfig.empirical_template: name -> (npz_path, key).
EMPIRICAL_REDSHIFT_TEMPLATES = {
    "SuperBIT": (
        "/home/habjan.e/TNG/Data/superbit_redshifts/redshift_arrays.npz",
        "selected_redshift",
    ),
    # All finite BPZ z_b over five LoVoCCS gen-2 clusters (A2384, A3571,
    # A3667, A3825, A3827); built by build_lovoccs_redshift_template.py.
    "LoVoCCS": (
        "/home/habjan.e/TNG/Data/lovoccs_redshifts/redshift_arrays.npz",
        "bpz_redshift",
    ),
}


@dataclass
class GridConfig:
    fov_mpc: float = 5.0                 # half-extent: everything spans [-fov_mpc, +fov_mpc]
    image_resolution: int = 16           # dynamics conditioning images (gal_xy, vz, vz_disp)
    cube_resolution: int = 16            # 3D density target cube
    vz_max_kms: float = 4000.0
    eps: float = 1e-12
    floor_value: float = -5.0            # value for empty / ultra-low voxels after normalization


@dataclass
class ScalingConfig:
    pos_mean: float = 0.0
    pos_std: float = 5.0
    vel_mean: float = 0.0
    vel_std: float = 800.0


@dataclass
class LensingConfig:
    # Snapshots are at z=0; each sample draws its lens redshift uniformly from
    # [z_lens_min, z_lens_max] for the lensing geometry.
    z_lens_min: float = 0.03
    z_lens_max: float = 0.12
    # LoVoCCS II (Fu+ 2024, arXiv:2402.10337): ~7 arcmin^-2 after photo-z cuts.
    n_source_per_arcmin2: float = 7.0
    sigma_e_per_component: float = 0.137
    # Source-galaxy grid: x/y are mapped to integer cell coordinates on an
    # (shape_grid_size x shape_grid_size) grid over [-fov_mpc, +fov_mpc], and
    # the Sigma -> kappa -> gamma maps are computed at this same resolution.
    # Written to the h5 attrs so the model knows how to sort sources into cells.
    shape_grid_size: int = 128
    # Which empirical n(z) to draw source redshifts from; one of
    # EMPIRICAL_REDSHIFT_TEMPLATES ('SuperBIT' or 'LoVoCCS').
    empirical_template: str = "LoVoCCS"
    # Resolved from empirical_template in __post_init__ when left empty;
    # set both explicitly to point at a custom npz instead.
    empirical_redshift_npz_path: str = ""
    empirical_redshift_key: str = ""
    r_inner_arcsec: float = 0.0          # inner core exclusion radius (0 = keep everything)
    use_magnification_bias: bool = True
    alpha_mag: float = 0.5               # source-magnitude function slope; 0.5 ~ mild enhancement
    mu_max: float = 10.0                 # cap to avoid critical-curve singularities

    def __post_init__(self):
        if not self.empirical_redshift_npz_path:
            if self.empirical_template not in EMPIRICAL_REDSHIFT_TEMPLATES:
                raise ValueError(
                    f"Unknown empirical_template={self.empirical_template!r}; "
                    f"expected one of {sorted(EMPIRICAL_REDSHIFT_TEMPLATES)}"
                )
            (self.empirical_redshift_npz_path,
             self.empirical_redshift_key) = EMPIRICAL_REDSHIFT_TEMPLATES[self.empirical_template]


@dataclass
class CosmologyConfig:
    H0: float = 70.0
    Om0: float = 0.3
    h_sim: float = 0.7                   # BAHAMAS h


@dataclass
class DatasetConfig:
    dataset_size: int = 10000
    test_fraction: float = 0.1           # fraction of clusters held out for validation
    rng_seed: int = 42
    simulations: Tuple[str, ...] = ("SIDM0.1b", "SIDM0.3b", "vdSIDMb", "CDMb")
    cluster_index_range: Tuple[int, int] = (1, 101)        # 1..100 inclusive
    sim_data_root: str = "/projects/mccleary_group/habjan.e/TNG/Data/"
    max_nodes: int = 700                 # cluster-member padded length


CFG_GRID = GridConfig()
CFG_SCALE = ScalingConfig()
CFG_LENS = LensingConfig()
CFG_COSMO = CosmologyConfig()
CFG_DATA = DatasetConfig()


# Smoke-test override: set SHAPE_DYN_SMOKE_TEST=1 to shrink the dataset to a
# handful of samples. Production defaults above are unchanged when unset.
import os as _os
if _os.environ.get("SHAPE_DYN_SMOKE_TEST", "0") == "1":
    CFG_DATA = DatasetConfig(
        dataset_size=12,
        test_fraction=1.0 / 3.0,                 # 1 of 3 clusters held out
        simulations=("CDMb",),
        cluster_index_range=(1, 4),              # 3 clusters
    )
