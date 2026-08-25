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
    # Member redshifts (the LOS feature/image channel). Measured over the four
    # simulations with z_l ~ U[0.03, 0.12]: <z_gal> = 0.0740, sigma = 0.0264,
    # the spread being dominated by the lens-redshift draw rather than by
    # cluster dynamics.
    z_mean: float = 0.075
    z_std: float = 0.026
    # Per-pixel redshift dispersion is ~z_std/13 (mean 0.0020 over the same
    # sample), so the dispersion channel gets its own scale.
    z_disp_std: float = 0.002


@dataclass
class LensingConfig:
    # Snapshots are at z=0; each sample draws its lens redshift uniformly from
    # [z_lens_min, z_lens_max] for the lensing geometry.
    z_lens_min: float = 0.03
    z_lens_max: float = 0.12
    # LoVoCCS II (Fu+ 2024, arXiv:2402.10337): ~7 arcmin^-2 after photo-z cuts.
    n_source_per_arcmin2: float = 7.0
    # LoVoCCS section 5.3: per-component shape dispersion ~0.4, against DES Y3's
    # ~0.26 (arXiv:2011.03408). This is the dispersion the *catalog realizes*;
    # intrinsic ellipticities are drawn from a wider Gaussian (~0.461, solved by
    # lensing_utils.gaussian_sigma_for_truncated) and truncated to the physical
    # |e| < 1, which removes the tail and narrows the distribution.
    sigma_e_realized: float = 0.40
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
    use_magnification_bias: bool = True
    alpha_mag: float = 0.5               # source-magnitude function slope; 0.5 ~ mild enhancement
    mu_max: float = 10.0                 # cap to avoid critical-curve singularities

    # --- survey masks -------------------------------------------------------
    # 1. Cluster members occult the sources behind them: a source landing on a
    #    member's light is blended and its shape unmeasurable. The BCG's disc
    #    also supersedes the old r_inner_arcsec core cut, on physical grounds.
    bcg_mask_radius_kpc: float = 100.0
    member_mask_radius_kpc: float = 30.0
    # 2. Sources are only measured where the telescope actually pointed. A
    #    rectangular footprint is drawn per sample, sides in arcmin (DECam is
    #    ~2.2 deg across, and LoVoCCS is a DECam survey), converted to Mpc at
    #    the lens redshift with lognormal scatter so the redshift dependence is
    #    noisy. It is then dithered n_exposures times: a source covered by every
    #    exposure is kept with probability 1, one covered by a single exposure
    #    with probability 1/n_exposures, and one outside the footprint never.
    #    Note the +/-5 Mpc field spans 4.7 deg at z=0.03 but only 1.3 deg at
    #    z=0.12, so a fixed angular footprint bites hard on nearby clusters and
    #    overfills the field for distant ones.
    pointing_side_arcmin: Tuple[float, float] = (100.0, 160.0)
    pointing_scale_scatter: float = 0.15      # lognormal sigma on the Mpc/arcmin scaling
    pointing_center_offset_frac: float = 0.05  # jitter of the field centre, in units of the side
    n_exposures: int = 10
    dither_amp_arcmin: Tuple[float, float] = (1.0, 3.0)
    coverage_exponent: float = 1.0            # P(keep) = coverage_fraction ** this

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
    # BAHAMAS cosmology (WMAP9; Robertson+ 2019, arXiv:1810.05649): Om0=0.2793,
    # OmL=0.7207, Ob=0.0463, h=0.7, sigma8=0.821, ns=0.972. Only Om0 is needed
    # here, for H(z_lens) when the members are placed in redshift space.
    Om0_sim: float = 0.2793


@dataclass
class XrayConfig:
    """
    Mock Chandra ACIS-I mosaics, ported from
    Sandbox_notebooks/mock_xray_maps/init_xray_map_code.ipynb.

    Built by build_xray_mosaics.py in a separate stage (pyXSIM/SOXS cost ~2 min
    per sample and several GB per worker), which caches one small npz per sample
    under scratch_dir; the dataset build only reads those back.
    """
    enabled: bool = True
    scratch_dir: str = "/scratch/habjan.e/xray_intermediates"
    # Photon lists: written once per (sim, cluster, z bin) by xray_photons.py and
    # read back by every sample of that cluster, so they persist for the whole
    # dataset build rather than being cleaned up per sample.
    photon_dir: str = "/scratch/habjan.e/xray_intermediates/photons"
    # Per-sample SOXS/pyXSIM products, deleted once the maps are extracted.
    work_dir: str = "/scratch/habjan.e/xray_intermediates/work"
    # Keep the full FITS/HDF5 intermediates for the first N samples only.
    keep_intermediates: int = 0

    # --- gas selection (notebook cells 12/16) -------------------------------
    n_R200: float = 2.0                  # load gas within this many R200 of the BCG
    T_min_K: float = 1.0e6               # X-ray emitting gas
    nH_max: float = 0.1                  # cm^-3, stands in for the missing SFR cut
    X_H: float = 0.76                    # primordial hydrogen mass fraction
    eps_grav_kpc: float = 5.7            # softening floor on the smoothing length
    Z_met: float = 0.3                   # Z_sun, no per-particle metals in the npz

    # --- photon sampling (notebook cells 24/26) -----------------------------
    emin_kev: float = 0.1
    emax_kev: float = 10.0
    nbins: int = 9900
    kT_min_kev: float = 0.086
    sample_area_cm2: float = 1000.0      # sampling area, > the 429 cm^2 ACIS-I peak ARF
    sample_exp_ks: float = 60.0          # photon-list exposure, > any observation below
    nH_gal: float = 0.03                 # 1e22 cm^-2 Galactic column
    absorb_model: str = "wabs"
    sky_center: Tuple[float, float] = (240.57, 15.97)   # Abell 2147, deg
    # Photon lists are expensive and bake in the redshift, so they are generated
    # per z bin of this width over [z_lens_min, z_lens_max] and reused across a
    # cluster's projections; the lensing geometry keeps the exact z_lens.
    z_bin_width: float = 0.01

    # --- the observation (notebook cells 38/42) -----------------------------
    instrument: str = "chandra_acisi_cy22"
    n_side: int = 3                      # 3x3 mosaic
    step_arcmin: float = 16.0            # < the 20' FOV, so pointings overlap
    t_exp_ks: float = 20.0               # per pointing
    band_emin_kev: float = 0.5
    band_emax_kev: float = 2.0
    reblock: int = 4                     # 2" mosaic pixels
    instr_bkgnd: bool = True
    foreground: bool = True
    ptsrc_bkgnd: bool = True

    # --- the stored maps ----------------------------------------------------
    # Fixed angular grid: the mosaic footprint is (n_side-1)*step + instrument
    # FOV = 52' across, so the physical scale varies with the sample's redshift
    # and is written per sample as xray_mpc_per_pix.
    grid_size: int = 128
    fov_arcmin: float = 52.0

    def pointing_offsets_deg(self):
        """Mosaic pointing offsets from sky_center, in degrees, as (dx, dy)."""
        import numpy as _np
        off = (_np.arange(self.n_side) - 0.5 * (self.n_side - 1)) * self.step_arcmin / 60.0
        return [(ddx, ddy) for ddy in off for ddx in off]

    # The redshift bins themselves live in xray_utils (z_bin_centers,
    # nearest_z_bin), so the photon stage and the dataset build cannot drift.


@dataclass
class DatasetConfig:
    # 5 simulations x 100 clusters x n_projections. The build writes a single
    # bahamas_dataset.hdf5; the train/val/test split happens later, across the
    # per-suite files rather than inside them.
    n_projections: int = 5
    rng_seed: int = 42
    simulations: Tuple[str, ...] = ("CDMb", "SIDM0.1b", "SIDM0.3b", "SIDM1b", "vdSIDMb")
    cluster_index_range: Tuple[int, int] = (1, 101)        # 1..100 inclusive
    sim_data_root: str = "/projects/mccleary_group/habjan.e/TNG/Data/"
    max_nodes: int = 700                 # cluster-member padded length

    # --- 'galaxy' selection -------------------------------------------------
    # BAHAMAS has no photometry, so the g-band cut is applied through a
    # selection function P(M_g <= mag_cut_g | Mstar) calibrated on TNG300,
    # which does have both (see mag_g_proxy_mask in the build script).
    tng_calibration_sims: Tuple[str, ...] = ("TNG300-2", "TNG300-3")
    tng_data_root: str = "/home/habjan.e/TNG/Data/TNG_data/"
    mag_cut_g: float = -18.0
    galaxy_cut_seed: int = 42            # fixed => the same members every projection
    # BCG search aperture, in units of R200 around the FoF centre of potential.
    # Without it the most massive stellar subhalo can be a neighbouring
    # cluster's BCG, which sits several Mpc away inside the 5 R200 cutout.
    bcg_search_radius_r200: float = 1.0


CFG_GRID = GridConfig()
CFG_SCALE = ScalingConfig()
CFG_LENS = LensingConfig()
CFG_COSMO = CosmologyConfig()
CFG_DATA = DatasetConfig()
CFG_XRAY = XrayConfig()


# Smoke-test override: set SHAPE_DYN_SMOKE_TEST=1 to shrink the dataset to a
# handful of samples. Production defaults above are unchanged when unset.
import os as _os
if _os.environ.get("SHAPE_DYN_SMOKE_TEST", "0") == "1":
    CFG_DATA = DatasetConfig(
        n_projections=4,
        simulations=("CDMb",),
        cluster_index_range=(1, 4),              # 3 clusters -> 12 samples
    )
