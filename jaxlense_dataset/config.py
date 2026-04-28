"""
Shared configuration for the jaxlense_dataset pipeline.

Pipeline stages (each has its own slurm script):
  Stage 1 (cl_dyn, CPU)     : build_kappa_truth_and_catalogs.py
  Stage 2 prep (cl_dyn, CPU): compute_kappa_power_spectrum.py
  Stage 2a (jax_lense, GPU) : train_score_for_clusters.py
  Stage 2b (jax_lense, GPU) : run_dlposterior.py
  Stage 3 (cl_dyn, CPU)     : assemble_dataset.py
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GridConfig:
    fov_mpc: float = 5.0                 # half-extent: FOV is [-fov_mpc, +fov_mpc]
    lens_recon_resolution: int = 128     # native resolution for kappa, gamma, n_gal_pix, score-net
    interim_cube_resolution: int = 64    # 3D cube saved at this resolution; downsampled at stage 3
    final_image_resolution: int = 16     # default; configurable in stage 3
    final_cube_resolution: int = 16      # default; configurable in stage 3
    eps: float = 1e-12
    floor_value: float = -5.0


@dataclass
class LensingConfig:
    # Snapshots are at z=0; assume the cluster lives at this redshift for lensing geometry.
    z_lens_assumed: float = 0.25
    n_source_per_arcmin2: float = 30.0
    sigma_e_per_component: float = 0.2
    # Smail+ 1994 n(z) ~ z^alpha exp(-(z/z0)^beta); these give <z>~1.05, median~0.94
    smail_alpha: float = 2.0
    smail_beta: float = 1.5
    smail_z0: float = 0.7
    smail_z_max: float = 5.0
    r_inner_arcsec: float = 30.0
    r_outer_arcmin: float = 10.0
    use_magnification_bias: bool = True
    alpha_mag: float = 0.5               # source-magnitude function slope; 0.5 ~ mild enhancement
    mu_max: float = 10.0                 # cap to avoid critical-curve singularities
    oversample_safety: float = 1.2       # extra buffer on Poisson draw before mag thinning


@dataclass
class CosmologyConfig:
    H0: float = 70.0
    Om0: float = 0.3
    h_sim: float = 0.7                   # BAHAMAS h (TNG: 0.6774; override via builder if needed)


@dataclass
class ScalingConfig:
    pos_mean: float = 0.0
    pos_std: float = 5.0
    vel_mean: float = 0.0
    vel_std: float = 800.0


@dataclass
class CatalogConfig:
    max_nodes: int = 700                 # cluster-member padded length, matching existing pipeline


@dataclass
class DatasetConfig:
    dataset_size: int = 10000
    test_fraction: float = 0.1
    rng_seed: int = 42
    simulations: Tuple[str, ...] = ("SIDM0.1b", "SIDM0.3b", "vdSIDMb", "CDMb")
    cluster_index_range: Tuple[int, int] = (1, 101)        # 1..100 inclusive
    sim_data_root: str = "/projects/mccleary_group/habjan.e/TNG/Data/"
    output_root: str = "/projects/mccleary_group/habjan.e/TNG/Data/jaxlense_dataset/"
    intermediate_dirname: str = "intermediate"
    posterior_dirname: str = "posterior"
    weights_dirname: str = "score_weights"
    final_dirname: str = "final"
    train_h5_template: str = "cond_diffusion_jaxlense_{img}img_{cube}cubed_train.h5"
    test_h5_template: str = "cond_diffusion_jaxlense_{img}img_{cube}cubed_test.h5"


@dataclass
class ScoreNetConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    training_steps: int = 30000
    noise_dist_std: float = 0.2
    spectral_norm: float = 1.0
    train_split_fraction: float = 0.9
    log_every: int = 50
    checkpoint_every: int = 5000


@dataclass
class DLPosteriorConfig:
    batch_size: int = 8                   # parallel chains per call
    n_independent_seeds: int = 4          # × batch_size = total samples per cluster
    initial_temperature: float = 0.15
    initial_step_size: float = 0.01
    min_steps_per_temp: int = 10
    num_steps_between_results: int = 6000
    cooling_gamma: float = 0.98
    min_temperature: float = 8e-3


CFG_GRID = GridConfig()
CFG_LENS = LensingConfig()
CFG_COSMO = CosmologyConfig()
CFG_SCALE = ScalingConfig()
CFG_CAT = CatalogConfig()
CFG_DATA = DatasetConfig()
CFG_SCORE = ScoreNetConfig()
CFG_DLP = DLPosteriorConfig()


# Smoke-test override: set JAXLENSE_SMOKE_TEST=1 to shrink the dataset to a
# handful of samples and reduce score-net / DLPosterior steps. Production
# defaults above are unchanged when the var is unset.
import os as _os
if _os.environ.get("JAXLENSE_SMOKE_TEST", "0") == "1":
    CFG_DATA = DatasetConfig(
        dataset_size=12,
        test_fraction=1.0 / 3.0,                 # 1 of 3 clusters held out
        simulations=("CDMb",),
        cluster_index_range=(1, 4),              # 3 clusters
    )
    CFG_SCORE = ScoreNetConfig(
        batch_size=4,
        training_steps=100,
        log_every=10,
        checkpoint_every=10**9,                  # disable mid-run checkpoints
    )
    CFG_DLP = DLPosteriorConfig(
        batch_size=2,
        n_independent_seeds=1,
        num_steps_between_results=400,
        min_temperature=0.05,
    )
