"""
Shared configuration for the shape_dynamics dataset pipeline.

Pipeline (each stage has its own slurm script):
  Prep (run once)      : build_lovoccs_redshift_template.py  (make_lovoccs_template.slurm)
  Dataset build        : build_shape_dynamics_dataset.py     (build_dataset.slurm)

Merges the pieces of jaxlense_dataset/config.py and
conditional_diffusion_model/conditional_diffusion_data.py that this
pipeline actually uses.

Simulation suites: the CFG_* objects below default to BAHAMAS. Setting the
environment variable CLUSTER_SUITE=FLAMINGO *before* this module is first
imported swaps CFG_DATA, CFG_COSMO, CFG_XRAY and CFG_DENS for their FLAMINGO versions
(see the bottom of the file); dataset/FLAMINGO/*.py set it themselves. Every
other setting -- grids, scalings, lensing, the X-ray source model and
observation -- is shared, so the two suites are built identically.
"""
import zlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np


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
    # Every resolution the dataset is built at. Each N gets its own N^3 density
    # cube and its own N x N dynamics products (images_N, gal_pixel_coords_N);
    # a model is trained on one N. The SPH field is deposited once at the
    # largest N and block-averaged down, so each N must divide the largest.
    resolutions: Tuple[int, ...] = (16, 32, 64, 128)
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
    # z_lens is the one random draw two separate slurm stages have to agree on:
    # xray_photons.py generates a photon list at each sample's z_lens, and
    # build_bahamas_dataset.py reads it back. It is therefore NOT taken from the
    # build's job-order rng stream (which would depend on the simulation list,
    # the cluster range and the loop order) but from sample_z_lens() below,
    # keyed only on (simulation, cluster, projection). Change this seed and
    # every photon list has to be regenerated.
    z_lens_seed: int = 20260907

    # --- photometric-redshift error (LoVoCCS II / III BPZ) ------------------
    # The empirical template *is* the observed BPZ point-estimate distribution
    # (real LoVoCCS z_b values), so a source's drawn z is what an observer
    # measures; the truth scatters about it. LoVoCCS III quotes
    # sigma_z/(1+z) ~ 0.051 with an outlier rate eta ~ 5.3%.
    #
    # Caveat, stated where it matters in draw_shape_catalog: 0.051 is the
    # scatter of z_phot about z_spec, so strictly it belongs to p(z_phot|z_true)
    # rather than p(z_true|z_phot). Inverting needs the true n(z) as a prior;
    # applying it symmetrically is the usual mock-catalog approximation.
    bpz_sigma_z: float = 0.051           # per (1 + z)
    bpz_outlier_rate: float = 0.053      # redrawn from the full n(z)
    apply_photoz_error: bool = True
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
    # LoVoCCS II's background-galaxy selection, applied to the BPZ point
    # estimate: a foreground cut at (cluster redshift + 0.1) and a high-z cut at
    # 1.4. Both are per sample, since the first depends on z_lens, and both are
    # applied to the *template* before drawing rather than to the drawn catalog,
    # so the surviving density is still n_source_per_arcmin2 -- LoVoCCS's
    # ~7 arcmin^-2 is itself quoted after these cuts.
    src_z_foreground_pad: float = 0.10
    src_z_max: float = 1.40
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
    # Photon lists: written once per (sim, cluster, projection) by
    # xray_photons.py and read back by the matching sample of the dataset build.
    # They are the expensive product of stage 1 -- hours of pyXSIM per cluster --
    # so they live on /projects rather than /scratch, which is subject to
    # purging, and are laid out one subdirectory per dark-matter model:
    #   <photon_dir>/<sim>/<sim>_GrNm_<cluster>_p<projection>_photons.h5
    photon_dir: str = "/projects/mccleary_group/habjan.e/TNG/Data/xray_list_files/BAHAMAS"
    # Scratch. Everything below is regenerable and gets deleted, so it stays on
    # the fast local filesystem.
    #   photon_work_dir : per-array-task cwd for stage 1, since pyXSIM writes
    #                     some intermediates relative to the working directory
    #   work_dir        : per-sample SOXS/pyXSIM products for stage 2, removed
    #                     once the maps are extracted
    photon_work_dir: str = "/scratch/habjan.e/xray_intermediates/photon_work"
    work_dir: str = "/scratch/habjan.e/xray_intermediates/work"

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
    # One photon list per *sample*, at that sample's exact z_lens
    # (config.sample_z_lens), rather than per redshift bin. Binning was the
    # cheaper option only in the abstract: n_projections = 5 lists per cluster
    # is fewer than the 9 bins the old 0.01-wide binning over [0.03, 0.12]
    # needed, and it removes the up-to-14% angular-scale mismatch a bin centre
    # left between a sample's X-ray map and its lensing geometry.

    # --- the observation ----------------------------------------------------
    # LoVoCCS's X-ray coverage is *archival* Chandra/XMM (LoVoCCS III), not a
    # uniform designed survey: a nearby cluster typically has a single ACIS
    # pointing of a few tens of ks, not a deep mosaic. So one pointing, with the
    # exposure drawn per sample from the archival-like distribution below.
    instrument: str = "chandra_acisi_cy22"
    n_side: int = 1                      # single pointing (was a 3x3 mosaic)
    step_arcmin: float = 16.0            # unused at n_side = 1; kept for mosaics
    band_emin_kev: float = 0.5
    band_emax_kev: float = 2.0
    reblock: int = 4                     # 2" detector pixels
    instr_bkgnd: bool = True
    foreground: bool = True
    ptsrc_bkgnd: bool = True

    # Per-sample exposure, lognormal about t_exp_median_ks and clipped to the
    # range. Typical archival ACIS exposures of nearby X-ray-luminous clusters
    # run from ~10 ks snapshots to ~100 ks pointed observations.
    #
    # The cap is not free: a pointing can only be as deep as the photon list it
    # samples, which needs sample_exp_ks * sample_area_cm2 >= t_exp * A_eff_peak.
    # With 60 ks x 1000 cm^2 against ACIS-I's 429 cm^2 peak ARF that allows
    # ~140 ks, so t_exp_max_ks = 100 leaves ~40% headroom.
    t_exp_median_ks: float = 30.0
    t_exp_sigma_ln: float = 0.6
    t_exp_min_ks: float = 10.0
    t_exp_max_ks: float = 100.0

    # --- the stored maps ----------------------------------------------------
    # Fixed *angular* grid, so a cluster's physical size varies with the
    # sample's redshift while the image shape does not; the scale is written per
    # sample as xray_mpc_per_pix. 20.008' is the ACIS-I field in SOXS's
    # instrument registry, so one pointing fills the frame.
    grid_size: int = 128
    fov_arcmin: float = 20.0

    # --- model-facing normalization -----------------------------------------
    # xray_mosaic and xray_counts are stored z-scored in log10 rather than raw:
    # both span ~4 decades with a large spike at zero, so a linear z-score would
    # leave 99% of pixels at ~0 and the core at +100 sigma.
    #
    # A pixel is usable only where the detector actually looked *and* a photon
    # actually landed. counts < min_counts also throws out the sub-count
    # reprojection leakage that soxs.make_mosaic_image leaves behind (~10% of
    # the field carries values of order 1e-26 counts, which log10 would send to
    # -26 and which are not photons); after the cut every surviving pixel has
    # counts >= 1, so log10 is finite everywhere it is evaluated.
    #
    # The two criteria are nearly redundant -- no exposure means no counts, so
    # the counts cut subsumes all but ~0.03% of the exposure cut -- but the
    # exposure cut is kept as the cheap statement of where the field is.
    min_counts: float = 0.5
    # Round numbers, not fitted: measured over 400 samples the valid-pixel log10
    # mosaic is (mean -7.97, std 0.486) and log10 counts is (mean 0.491, std
    # 0.487). Held fixed so the constants do not move when other simulation
    # suites join the split.
    mosaic_log10_mean: float = -8.0
    mosaic_log10_std: float = 0.5
    counts_log10_mean: float = 0.5
    counts_log10_std: float = 0.5

    # --- intrinsic X-ray luminosity (diagnostic, never a cut) ---------------
    # LoVoCCS selects on L_X500c > 1e44 erg/s in 0.1-2.4 keV. The BAHAMAS sample
    # is mass-selected, so L_X500c is recorded per cluster to let that selection
    # be checked after the fact. No simulated cluster is dropped on it.
    lx_band_kev: Tuple[float, float] = (0.1, 2.4)
    lx_nbins: int = 4000
    lx_T_grid_min_K: float = 1.0e6       # matches T_min_K; Lambda = 0 below
    lx_T_grid_max_K: float = 1.0e9
    lx_T_grid_n: int = 300

    def pointing_offsets_deg(self):
        """Pointing offsets from sky_center, in degrees, as (dx, dy)."""
        import numpy as _np
        off = (_np.arange(self.n_side) - 0.5 * (self.n_side - 1)) * self.step_arcmin / 60.0
        return [(ddx, ddy) for ddy in off for ddx in off]

    def draw_exposure_ks(self, rng):
        """Per-sample ACIS exposure, in ks."""
        import numpy as _np
        t = float(rng.lognormal(_np.log(self.t_exp_median_ks), self.t_exp_sigma_ln))
        return float(_np.clip(t, self.t_exp_min_ks, self.t_exp_max_ks))

    # The redshift each photon list is generated at lives in config.sample_z_lens,
    # so the photon stage and the dataset build cannot drift.

    # The redshift bins themselves live in xray_utils (z_bin_centers,
    # nearest_z_bin), so the photon stage and the dataset build cannot drift.


@dataclass
class DensityConfig:
    """
    Adaptive-SPH density cubes (density_utils.py), chosen against NGP, fixed
    Gaussian, DTFE and 32/128-neighbour SPH in
    /scratch/habjan.e/density_smoothing_tests.
    """
    n_ngb: int = 64                      # h_j = distance to the n_ngb-th neighbour, all species
    # h_j is floored at 2.8 times the simulation's Plummer-equivalent softening
    # (Gadget's spline support), from SOFTENING_KPC via sph_h_floor_mpc(sim).
    sub: float = 6.0                     # sub-samples across a kernel's support
    # Per-cluster h_j caches (projection-independent), on scratch since they
    # regenerate: <cache_dir>/<sim>/<stem>_<idx>_ngb<n_ngb>.npz. The build's
    # raw-cube side file goes here too.
    cache_dir: str = "/scratch/habjan.e/cube_intermediates/BAHAMAS"


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
    # Per-cluster files are <sim_data_root>/<sim>/<cluster_file_stem>_<idx:03d>.npz;
    # the stem also names the photon lists (xray_utils.photon_prefix).
    cluster_file_stem: str = "GrNm"
    # Cluster-member padded length, shared by every suite so the prior spans
    # the full range of cluster sizes. Members beyond max_nodes would be
    # dropped in catalogue order, so it sits above the largest count passing
    # the g-band cut anywhere: 2603 (FLAMINGO L2p8_m9 cluster_000, M200 =
    # 5.05e15). BAHAMAS peaks at 604, the other FLAMINGO runs at 1691 (Jet).
    max_nodes: int = 2650
    # Total sample count to reach when it is not a multiple of the cluster
    # count (0 = exactly n_projections per cluster). Then every cluster gets
    # n_projections and n_samples_target - n_clusters * n_projections of them,
    # chosen per simulation by projection_seed, get one more; see
    # n_projections_for(), which both slurm stages call.
    n_samples_target: int = 0
    projection_seed: int = 20261012

    # --- 'galaxy' selection -------------------------------------------------
    # BAHAMAS has no photometry, so the g-band cut is applied through a
    # selection function P(M_g <= mag_cut_g | Mstar) calibrated on TNG300,
    # which does have both (see mag_g_proxy_mask in the build script).
    # All three TNG300 resolutions, for every suite (BAHAMAS, FLAMINGO, TNG300).
    tng_calibration_sims: Tuple[str, ...] = ("TNG300-1", "TNG300-2", "TNG300-3")
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
CFG_DENS = DensityConfig()


# ============================================================
# Simulation suites
# ============================================================

# FLAMINGO runs with per-particle cutouts under FLAMINGO_DATA_ROOT, excluding
# NoCooling: a non-radiative run forms no stars and no black holes, so it has
# no BCG to centre on and no stellar masses to select members with.
FLAMINGO_SIMULATIONS = (
    "L1_m9", "fgas+2sigma", "fgas-2sigma", "Jet", "Mstar-1sigma",
    "Planck", "PlanckNu0p24Fix", "PlanckNu0p24Var", "PlanckNu0p48Fix",
    "L2p8_m9",
)
FLAMINGO_DATA_ROOT = "/projects/mccleary_group/habjan.e/FLAMINGO/"

# (h, Om0) per FLAMINGO run. Om0 from https://flamingo.strw.leidenuniv.nl/
# simulations.html (Schaye+ 2023 Table 4, Elbers+ 2024 Table 1); h is the
# value stored in the cutouts, which carries one more digit than the page
# (0.6732 against 0.673). Om0 includes the massive neutrinos, and the runs are
# flat, so Omega_Lambda = 1 - Om0 as for BAHAMAS.
_D3A = (0.681, 0.306)
FLAMINGO_COSMOLOGY = {
    "L1_m9": _D3A, "L2p8_m9": _D3A, "fgas+2sigma": _D3A, "fgas-2sigma": _D3A,
    "Jet": _D3A, "Mstar-1sigma": _D3A, "NoCooling": _D3A,
    "Planck": (0.6732, 0.316),
    "PlanckNu0p24Fix": (0.6732, 0.316),
    "PlanckNu0p24Var": (0.6623, 0.328),
    "PlanckNu0p48Fix": (0.6732, 0.316),
}

# TNG300: halos 0-99 of each resolution, from the 5r200 cutouts in
# <TNG300_DATA_ROOT>/5r200_data-{1,2,3} (read by TNG300/tng300_io.py). All
# three share the Planck 2015 cosmology of the snapshot headers.
TNG300_SIMULATIONS = ("TNG300-1", "TNG300-2", "TNG300-3")
TNG300_DATA_ROOT = "/projects/mccleary_group/habjan.e/TNG/Data/TNG_data/"
TNG300_COSMOLOGY = {s: (0.6774, 0.3089) for s in TNG300_SIMULATIONS}

# Physical Plummer-equivalent gravitational softening at z = 0, kpc, of the
# dark matter and stars: the floor of the SPH smoothing lengths
# (sph_h_floor_mpc). BAHAMAS 4 kpc/h (Robertson+ 2019); FLAMINGO m9 2.85 kpc
# (Schaye+ 2023); TNG300 1/2/4 ckpc/h at h = 0.6774 (Pillepich+ 2018,
# Nelson+ 2019).
BAHAMAS_SOFTENING_KPC = 5.7
SOFTENING_KPC = {
    **{s: 2.85 for s in FLAMINGO_SIMULATIONS},
    "TNG300-1": 1.48, "TNG300-2": 2.95, "TNG300-3": 5.90,
}


def sph_h_floor_mpc(sim):
    """Floor of the SPH smoothing lengths for `sim`: 2.8 x softening, in Mpc."""
    return 2.8 * SOFTENING_KPC.get(sim, BAHAMAS_SOFTENING_KPC) / 1e3


import math as _math
import os as _os

SUITE = _os.environ.get("CLUSTER_SUITE", "BAHAMAS")
if SUITE == "FLAMINGO":
    # One projection per cluster: 10 runs x 250 clusters = 2500 samples.
    CFG_DATA = DatasetConfig(
        n_projections=1,
        simulations=FLAMINGO_SIMULATIONS,
        cluster_index_range=(0, 250),            # cluster_000..cluster_249
        sim_data_root=FLAMINGO_DATA_ROOT,
        cluster_file_stem="cluster",
    )
    # The cosmology differs between FLAMINGO runs, so there is no suite-wide
    # h_sim / Om0_sim: everything goes through sim_cosmology(sim). NaN makes
    # any leftover use of the suite-wide values fail loudly.
    CFG_COSMO = CosmologyConfig(h_sim=float("nan"), Om0_sim=float("nan"))
    CFG_XRAY = XrayConfig(
        photon_dir="/projects/mccleary_group/habjan.e/TNG/Data/xray_list_files/FLAMINGO",
        photon_work_dir="/scratch/habjan.e/xray_intermediates/photon_work_flamingo",
        work_dir="/scratch/habjan.e/xray_intermediates/work_flamingo",
    )
    CFG_DENS = DensityConfig(cache_dir="/scratch/habjan.e/cube_intermediates/FLAMINGO")
elif SUITE == "TNG300":
    # 3 resolutions x 100 halos = 300 clusters, 8 projections each (2400) plus
    # a 9th for 100 of them -> 2500 samples, the size of the other two files.
    CFG_DATA = DatasetConfig(
        n_projections=8,
        n_samples_target=2500,
        simulations=TNG300_SIMULATIONS,
        cluster_index_range=(0, 100),            # halos 0..99
        sim_data_root=TNG300_DATA_ROOT,
        cluster_file_stem="halo",
    )
    # Same cosmology for all three, but routed through sim_cosmology like
    # FLAMINGO so the build code is identical; NaN catches any leftover use.
    CFG_COSMO = CosmologyConfig(h_sim=float("nan"), Om0_sim=float("nan"))
    CFG_XRAY = XrayConfig(
        photon_dir="/projects/mccleary_group/habjan.e/TNG/Data/xray_list_files/TNG300",
        photon_work_dir="/scratch/habjan.e/xray_intermediates/photon_work_tng300",
        work_dir="/scratch/habjan.e/xray_intermediates/work_tng300",
    )
    CFG_DENS = DensityConfig(cache_dir="/scratch/habjan.e/cube_intermediates/TNG300")
elif SUITE != "BAHAMAS":
    raise ValueError(f"CLUSTER_SUITE={SUITE!r}; expected 'BAHAMAS', 'FLAMINGO' or 'TNG300'")


def sim_cosmology(sim):
    """
    (h, Om0) of one simulation: per run for FLAMINGO and TNG300, CFG_COSMO's
    BAHAMAS values otherwise. Used for H(z) in the members' redshift-space
    placement and for the X-ray maps' angular scale, so both match the
    simulation.
    """
    if sim in FLAMINGO_COSMOLOGY:
        return FLAMINGO_COSMOLOGY[sim]
    if sim in TNG300_COSMOLOGY:
        return TNG300_COSMOLOGY[sim]
    h, Om0 = float(CFG_COSMO.h_sim), float(CFG_COSMO.Om0_sim)
    if not (_math.isfinite(h) and _math.isfinite(Om0)):
        raise KeyError(f"no cosmology for simulation {sim!r} in suite {SUITE}")
    return h, Om0


# BAHAMAS box (cMpc/h). The BAHAMAS npz files carry no box size; the FLAMINGO
# loader (FLAMINGO/flamingo_io.py) adds one under BOXSIZE_KEY, already in cMpc/h.
BAHAMAS_BOXSIZE_MPC_OVER_H = 400.0
BOXSIZE_KEY = "boxsize_mpc_over_h"


def cluster_boxsize_mpc_over_h(data):
    """Periodic box size of a loaded cluster file, in comoving Mpc/h."""
    if BOXSIZE_KEY in data.files:
        return float(data[BOXSIZE_KEY])
    return BAHAMAS_BOXSIZE_MPC_OVER_H


# ============================================================
# The one draw the two slurm stages share
# ============================================================

def sample_z_lens(sim, cluster_idx, proj_index, cfg=CFG_LENS):
    """
    The lens redshift of sample (sim, cluster_idx, proj_index), drawn uniformly
    from [z_lens_min, z_lens_max].

    Both stages call this and must get the same answer: xray_photons.py builds a
    pyXSIM photon list at exactly this redshift, and build_bahamas_dataset.py
    reads that list back for the matching sample. So the draw is keyed on the
    sample's identity via SeedSequence rather than pulled from a sequential rng
    stream -- it does not move if the simulation tuple, the cluster range, the
    projection count or the loop order changes, and either stage can compute it
    for one sample without replaying the others.

    crc32 rather than hash() for the simulation name: Python string hashing is
    salted per process, so hash() would give a different answer on every run.
    """
    key = [int(cfg.z_lens_seed),
           int(zlib.crc32(str(sim).encode("utf-8"))),
           int(cluster_idx),
           int(proj_index)]
    rng = np.random.default_rng(np.random.SeedSequence(key))
    return float(rng.uniform(cfg.z_lens_min, cfg.z_lens_max))


_EXTRA_PROJECTIONS = {}


def _extra_projection_clusters(cfg):
    """
    The (sim, cluster) pairs that get one projection beyond cfg.n_projections,
    so the suite totals cfg.n_samples_target. The extras are split as evenly
    as possible across simulations (100 over 3 -> 33/33/34) and drawn within
    each by a SeedSequence keyed on projection_seed and the simulation name,
    so the choice is fixed by the config alone.
    """
    key = (cfg.n_projections, cfg.n_samples_target, cfg.projection_seed,
           tuple(cfg.simulations), tuple(cfg.cluster_index_range))
    if key in _EXTRA_PROJECTIONS:
        return _EXTRA_PROJECTIONS[key]
    clusters = list(range(*cfg.cluster_index_range))
    n_sims = len(cfg.simulations)
    n_extra = cfg.n_samples_target - cfg.n_projections * n_sims * len(clusters)
    if not 0 <= n_extra <= n_sims * len(clusters):
        raise ValueError(
            f"n_samples_target={cfg.n_samples_target} is not reachable with "
            f"{cfg.n_projections} or {cfg.n_projections + 1} projections of "
            f"{n_sims} x {len(clusters)} clusters")
    per_sim, rem = divmod(n_extra, n_sims)
    chosen = set()
    for i, sim in enumerate(cfg.simulations):
        n = per_sim + (1 if i >= n_sims - rem else 0)     # remainder to the last sims
        rng = np.random.default_rng(np.random.SeedSequence(
            [int(cfg.projection_seed), int(zlib.crc32(str(sim).encode("utf-8")))]))
        for c in rng.choice(clusters, size=n, replace=False):
            chosen.add((str(sim), int(c)))
    _EXTRA_PROJECTIONS[key] = chosen
    return chosen


def n_projections_for(sim, cluster_idx, cfg=None):
    """
    How many projections (sample_z_lens proj_index 0..n-1) cluster
    `cluster_idx` of `sim` has. cfg.n_projections for every cluster unless
    cfg.n_samples_target asks for more; then a fixed subset gets one extra
    (_extra_projection_clusters). The photon stage and the dataset build both
    call this, so they agree on the sample list without sharing any state.
    """
    cfg = CFG_DATA if cfg is None else cfg
    if not cfg.n_samples_target:
        return int(cfg.n_projections)
    extra = (str(sim), int(cluster_idx)) in _extra_projection_clusters(cfg)
    return int(cfg.n_projections) + int(extra)


# Smoke-test override: set SHAPE_DYN_SMOKE_TEST=1 to shrink the dataset to a
# handful of samples. Production defaults above are unchanged when unset.
if _os.environ.get("SHAPE_DYN_SMOKE_TEST", "0") == "1":
    if SUITE == "FLAMINGO":
        import dataclasses as _dc
        CFG_DATA = _dc.replace(
            CFG_DATA,
            simulations=("L1_m9",),
            cluster_index_range=(0, 3),          # the 3 most massive -> 3 samples
        )
    elif SUITE == "TNG300":
        import dataclasses as _dc
        CFG_DATA = _dc.replace(
            CFG_DATA,
            simulations=("TNG300-1",),
            cluster_index_range=(0, 1),          # halo 0, the most massive
            n_projections=1,
            n_samples_target=0,                  # -> 1 sample
        )
    else:
        CFG_DATA = DatasetConfig(
            n_projections=4,
            simulations=("CDMb",),
            cluster_index_range=(1, 4),          # 3 clusters -> 12 samples
        )
