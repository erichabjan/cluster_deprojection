#!/usr/bin/env python3
"""
Build the BAHAMAS dataset for the conditional diffusion model.

Writes a single bahamas_dataset.hdf5 covering every (simulation, cluster,
projection). It holds no train/val split: the split is made later, across the
per-simulation-suite files, once the other suites have been run through the same
analysis.

Per (cluster, sim, projection) sample, all in ONE viewing frame (a single
projection vector rotates both the cluster members and the matter particles),
with every resolution-dependent product stored once per N in
CFG_GRID.resolutions under an _N suffix (a model trains on one N):
  - dynamics conditioning images images_N (3, N, N): gal_xy, gal_z_xy, gal_z_disp_xy
  - cluster-member point cloud, padded to max_nodes (features/targets/mask), with
    gal_pixel_coords_N on the N x N image grid
  - 3D density target cubes density_cube_N (N, N, N), log-standardized with this
    file's per-N stats, + global targets (from the finest cube, shared by all N).
    The density is adaptive SPH (density_utils.py): a cubic-spline kernel per
    particle with h_j the distance to its CFG_DENS.n_ngb-th neighbour, deposited
    once at the largest N and block-averaged down, so every N is the same field
    integrated over a different voxel size.
  - NEW: weak-lensing source-galaxy point cloud replacing the projected mass image.
    A lens redshift z_l ~ U[z_lens_min, z_lens_max] is drawn per sample; the rotated
    matter particles give Sigma -> kappa_inf -> gamma (KS93) on a
    (shape_grid_size x shape_grid_size) grid over [-fov_mpc, +fov_mpc]; then
    draw_shape_catalog draws sources (LoVoCCS n(z), n_source_per_arcmin2, intrinsic
    ellipticity, magnification bias) and shears them.

    The cluster is centred on its BCG (the most massive stellar subhalo within
    bcg_search_radius_r200 * R200 of the FoF centre of potential) rather than on
    the centre of potential itself, members are selected with a TNG-calibrated
    g-band proxy cut, and their LOS velocities are stored as redshifts: the BCG
    sits at z_l and every other member is placed in redshift space relative to
    it, using peculiar velocity + Hubble flow in the BAHAMAS cosmology.

    Two survey masks then cull that catalog: cluster members occult the sources
    behind them (100 kpc for the BCG, 30 kpc for the rest), and only sources
    inside a dithered rectangular pointing survive, with probability equal to
    the fraction of exposures covering them.

    Sources follow LoVoCCS II's background selection (z_lens + 0.1 < z_phot
    < 1.4, applied to the n(z) template before drawing so the density stays at
    ~7 arcmin^-2) and carry LoVoCCS III's BPZ error: the lensing uses a true
    redshift scattered about z_phot by sigma_z/(1+z) = 0.051 with a 5.3%
    catastrophic-outlier rate, while the catalog reports only z_phot. Sources
    scattered in front of the lens are unsheared, which is the contamination a
    real cluster WL analysis has to correct for; it is summarised per sample as
    src_true_foreground_frac.

    Each sample stores, ragged (no padding, variable N):
      src_features (N, 8) float32: e1, e2, ix, iy, z_s, z_lens, cell_frac, coverage
        - z_s: the BPZ point estimate, i.e. what an observer measures
        - ix, iy: integer cell coordinates on the shape grid (stored as float32)
        - cell_frac: (# sources in that cell) / N
        - coverage: fraction of exposures covering that source
      src_cell_id  (N,)   int32: iy * shape_grid_size + ix; rows are PRE-SORTED by this
      src_cell_ptr (S*S+1,) int64: CSR pointers, cell c's sources are rows ptr[c]:ptr[c+1]
      src_coverage (S, S)  float32: the same coverage fraction on the shape grid,
        so an empty cell is distinguishable from an unobserved one

  - NEW: a mock Chandra ACIS-I X-ray observation of the same cluster, in the
    same viewing frame. xray_photons.py has already made one pyXSIM photon list
    per (sim, cluster, projection), at that sample's exact z_lens; each sample
    projects its own list along its viewing direction and observes it. The two
    stages agree on z_lens through config.sample_z_lens, which keys the draw on
    the sample's identity instead of on either script's rng stream.

    LoVoCCS's X-ray coverage is archival Chandra/XMM rather than a designed
    survey, so a cluster gets one ACIS-I pointing at a depth drawn per sample
    (lognormal about 30 ks, clipped to 10-100 ks) instead of a uniform mosaic.
    Stored on a fixed 20' *angular* grid -- one ACIS-I field -- so a cluster's
    apparent size varies with its redshift while the image shape does not; the
    physical scale is a per-sample attribute (xray_mpc_per_pix):
      xray_mosaic (G,G): exposure-corrected 0.5-2 keV surface brightness
      xray_counts (G,G): counts
      xray_expmap (G,G): mean exposure map, cm^2 (SOXS normalises out the time)
      xray_ideal  (G,G): the same band with no instrument, PSF or background
      xray_mask   (G,G): 1.0 where the pixel is usable, 0.0 where it is not
    plus the group attrs xray_t_exp_ks (the realised depth) and lx500c_erg_s
    (intrinsic 0.1-2.4 keV luminosity inside R500c, for comparison against
    LoVoCCS's L_X > 1e44 erg/s selection -- a diagnostic, never a cut).

    The two model-facing maps are stored ready to train on rather than raw.
    xray_mask marks the pixels that carry information -- the detector looked
    there (xray_expmap > 0) and a photon landed (counts > CFG_XRAY.min_counts,
    which also drops the sub-count reprojection leakage soxs leaves over ~10%
    of the field) -- and on those pixels xray_mosaic and xray_counts hold
    log10 z-scores against the fixed constants in CFG_XRAY, 0 elsewhere. Both
    span ~4 decades over a large spike at zero, so a linear z-score would leave
    99% of pixels at ~0 and the core at +100 sigma. The raw values invert from
    the stored ones (see the xray_standardization file attr), so they are not
    written twice; xray_expmap and xray_ideal stay in physical units.

Prerequisites:
  - build_lovoccs_redshift_template.py (run once) when
    CFG_LENS.empirical_template = 'LoVoCCS';
  - xray_photons.slurm, unless CFG_XRAY.enabled is False. The build checks every
    photon list it will need up front and refuses to start if any are missing.

Run: cl_dyn env, multiple CPUs (build_bahamas_dataset.slurm).
"""
import os
import sys
import h5py
import numpy as np
import multiprocessing as mp
from typing import Dict, Tuple

from sklearn.isotonic import IsotonicRegression

dirc_path = "/home/habjan.e/"
sys.path.append(dirc_path + "TNG/TNG_cluster_dynamics")
import TNG_DA  # rotate_to_viewing_frame

# config.py, lensing_utils.py and xray_utils.py are shared across simulations and
# live one level up, in dataset/.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (CFG_GRID, CFG_SCALE, CFG_LENS, CFG_COSMO, CFG_DATA,
                    CFG_XRAY, CFG_DENS, sph_h_floor_mpc, sample_z_lens)
import lensing_utils as lu
import density_utils as du

# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------
OUT_DIR = "/projects/mccleary_group/habjan.e/TNG/Data/shape_dynamics/"
# One file per simulation suite. BAHAMAS is the first; the train/val/test split
# is made across these files later, not within them.
OUT_FILENAME = "bahamas_dataset.hdf5"

BOXSIZE_MPC_OVER_H = 400.0   # BAHAMAS box (cMpc/h)
EPS = 1e-9
C_KMS = 299792.458           # speed of light, km/s

# Threads per worker for the SPH kNN and deposit; the pool gets
# SLURM_CPUS_PER_TASK // SPH_THREADS workers. 1 unless memory, not CPUs, caps
# the worker count (TNG300-1: ~10^8 particles per cluster).
SPH_THREADS = int(os.environ.get("SPH_THREADS", "1"))

# Gaussian width whose |e| < 1 truncation realizes CFG_LENS.sigma_e_realized.
SIGMA_E_INPUT = lu.gaussian_sigma_for_truncated(CFG_LENS.sigma_e_realized)

# Populated by _init_worker: empirical n(z) template, foreground-cut above z_lens_max.
_EMPIRICAL_Z_TEMPLATE = None
# Populated by _init_worker: (TNG stellar masses [Msun], TNG g-band magnitudes)
# used to calibrate the galaxy selection function.
_MAG_CALIBRATION = None


def _init_worker(z_template, mag_calibration):
    global _EMPIRICAL_Z_TEMPLATE, _MAG_CALIBRATION
    _EMPIRICAL_Z_TEMPLATE = z_template
    _MAG_CALIBRATION = mag_calibration
    if CFG_XRAY.enabled:
        # pyXSIM and SOXS write several products relative to the working
        # directory, so give every worker the same scratch area to litter.
        os.makedirs(CFG_XRAY.work_dir, exist_ok=True)
        os.chdir(CFG_XRAY.work_dir)


# ============================================================
# 'Galaxy' selection (from figures/paper_plots.ipynb)
# ============================================================

def load_tng_calibration():
    """
    TNG300 stellar masses (Msun) and g-band absolute magnitudes, concatenated
    over CFG_DATA.tng_calibration_sims. Read from the local hdf5 caches under
    CFG_DATA.tng_data_root; only re-downloaded if one is missing.
    """
    sys.path.append(dirc_path + "TNG/Codes/TNG_workshop")
    import iapi_TNG as iapi

    mstar, mag_g = [], []
    for sim in CFG_DATA.tng_calibration_sims:
        stem = os.path.join(CFG_DATA.tng_data_root, sim + "_")
        mass_type = iapi.getSubhaloField(
            "SubhaloMassType", simulation=sim, snapshot=99,
            fileName=stem + "SubhaloMassType", rewriteFile=0)
        photometrics = iapi.getSubhaloField(
            "SubhaloStellarPhotometrics", simulation=sim, snapshot=99,
            fileName=stem + "SubhaloStellarPhotometrics", rewriteFile=0)
        mstar.append(mass_type[:, 4])
        mag_g.append(photometrics[:, 4])

    # SubhaloMassType is in 1e10 Msun/h (TNG300 h = 0.6774); BAHAMAS
    # sub_massTotal is already in Msun, so the calibration has to be converted.
    mstar = np.concatenate(mstar).astype(np.float64) * 1e10 / 0.6774
    mag_g = np.concatenate(mag_g).astype(np.float64)
    return mstar, mag_g


def mag_g_proxy_mask(
    mstar_cal,
    mag_g_cal,
    mstar_target,
    mag_cut=-18.0,
    logmass_bins=None,
    min_bin_count=30,
    method="stochastic",
    completeness_level=0.90,
    seed=None,
    return_info=False,
):
    """
    Select subhalos in a simulation without g-band magnitudes using a
    selection function calibrated from a simulation with magnitudes.

    Parameters
    ----------
    mstar_cal : array-like
        Stellar masses in the calibration simulation.
    mag_g_cal : array-like
        Corresponding g-band absolute magnitudes.
    mstar_target : array-like
        Stellar masses in the simulation to which the selection is applied.
    mag_cut : float, default=-18
        Magnitude threshold. Objects with mag_g <= mag_cut pass.
    logmass_bins : array-like, optional
        Bin edges in log10(stellar mass). By default, 0.15-dex bins are used.
    min_bin_count : int, default=30
        Minimum number of calibration objects required in a mass bin.
    method : {"stochastic", "hard"}
        "stochastic": retain each target object with probability
        P(mag_g <= mag_cut | Mstar).

        "hard": retain all target objects above the stellar mass where the
        fitted selection probability reaches `completeness_level`.
    completeness_level : float, default=0.90
        Completeness used when method="hard".
    seed : int, optional
        Random seed for reproducible stochastic masks.
    return_info : bool, default=False
        If True, also return selection probabilities and calibration details.

    Returns
    -------
    mask : ndarray of bool
        Boolean selection mask for `mstar_target`.

    info : dict, optional
        Returned only when `return_info=True`.
    """
    mstar_cal = np.asarray(mstar_cal, dtype=float)
    mag_g_cal = np.asarray(mag_g_cal, dtype=float)
    mstar_target = np.asarray(mstar_target, dtype=float)

    if mstar_cal.shape != mag_g_cal.shape:
        raise ValueError("mstar_cal and mag_g_cal must have the same shape.")

    if method not in {"stochastic", "hard"}:
        raise ValueError("method must be 'stochastic' or 'hard'.")

    if not 0 < completeness_level <= 1:
        raise ValueError("completeness_level must be between 0 and 1.")

    # Remove invalid calibration entries.
    valid_cal = (
        np.isfinite(mstar_cal)
        & np.isfinite(mag_g_cal)
        & (mstar_cal > 0)
    )

    logm_cal = np.log10(mstar_cal[valid_cal])
    passes_mag_cut = mag_g_cal[valid_cal] <= mag_cut

    if logmass_bins is None:
        lower = np.floor(logm_cal.min() / 0.15) * 0.15
        upper = np.ceil(logm_cal.max() / 0.15) * 0.15 + 0.15
        logmass_bins = np.arange(lower, upper, 0.15)
    else:
        logmass_bins = np.asarray(logmass_bins, dtype=float)

    bin_centers = 0.5 * (logmass_bins[:-1] + logmass_bins[1:])

    n_all, _ = np.histogram(logm_cal, bins=logmass_bins)
    n_pass, _ = np.histogram(logm_cal[passes_mag_cut], bins=logmass_bins)

    raw_completeness = np.divide(
        n_pass,
        n_all,
        out=np.full(n_all.shape, np.nan, dtype=float),
        where=n_all > 0,
    )

    usable = (n_all >= min_bin_count) & np.isfinite(raw_completeness)

    if usable.sum() < 2:
        raise ValueError(
            "Too few populated stellar-mass bins to calibrate the selection."
        )

    calibration_mass = bin_centers[usable]

    # Enforce the physically expected non-decreasing completeness with mass.
    isotonic = IsotonicRegression(
        increasing=True,
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    fitted_completeness = isotonic.fit_transform(
        calibration_mass,
        raw_completeness[usable],
        sample_weight=n_all[usable],
    )

    # Calculate selection probabilities for the target simulation.
    target_valid = np.isfinite(mstar_target) & (mstar_target > 0)
    probabilities = np.zeros(mstar_target.shape, dtype=float)

    probabilities[target_valid] = np.interp(
        np.log10(mstar_target[target_valid]),
        calibration_mass,
        fitted_completeness,
        left=0.0,
        right=1.0,
    )

    mass_threshold = None

    if method == "stochastic":
        rng = np.random.default_rng(seed)
        mask = target_valid & (
            rng.random(mstar_target.shape) < probabilities
        )

    else:
        reaches_level = fitted_completeness >= completeness_level

        if not np.any(reaches_level):
            raise ValueError(
                f"The fitted selection never reaches "
                f"{completeness_level:.0%} completeness."
            )

        first = np.flatnonzero(reaches_level)[0]

        # Interpolate the mass at which completeness crosses the requested level.
        if first == 0:
            logmass_threshold = calibration_mass[0]
        else:
            logmass_threshold = np.interp(
                completeness_level,
                fitted_completeness[first - 1:first + 1],
                calibration_mass[first - 1:first + 1],
            )

        mass_threshold = 10**logmass_threshold
        mask = target_valid & (mstar_target >= mass_threshold)

    if return_info:
        info = {
            "probability": probabilities,
            "bin_centers_log10": bin_centers,
            "n_all": n_all,
            "n_pass": n_pass,
            "raw_completeness": raw_completeness,
            "calibration_logmass": calibration_mass,
            "fitted_completeness": fitted_completeness,
            "mass_threshold": mass_threshold,
        }
        return mask, info

    return mask


# ============================================================
# Redshift-space placement of the members
# ============================================================

def hubble_kms_mpc(z):
    """H(z) in km/s/Mpc for the BAHAMAS (WMAP9) cosmology."""
    H0 = 100.0 * CFG_COSMO.h_sim
    return H0 * np.sqrt(CFG_COSMO.Om0_sim * (1.0 + z) ** 3
                        + (1.0 - CFG_COSMO.Om0_sim))


def members_to_redshift(z_los_mpc, v_los_kms, z_bcg):
    """
    Observed redshift of each member, with the BCG at z_bcg and at the origin
    of the viewing frame. The LOS offset contributes Hubble flow and the
    peculiar velocity (already in the BCG rest frame) the Doppler shift:

        v_tot = v_los + H(z_bcg) * z_los,
        z_gal = z_bcg + (1 + z_bcg) * v_tot / c.
    """
    v_tot = v_los_kms + hubble_kms_mpc(z_bcg) * z_los_mpc
    return z_bcg + (1.0 + z_bcg) * v_tot / C_KMS


# ============================================================
# Dynamics helpers (from conditional_diffusion_data.py)
# ============================================================

def make_galaxy_map_xy(x, y, fov, N_img):
    n = max(len(x), 1)
    w = np.full_like(x, 1.0 / n, dtype=np.float32)
    return lu.bin2d_sum(x, y, w, fov, N_img)


def make_galaxy_los_mean_xy(x, y, s_los, fov, N_img):
    vz_sum = lu.bin2d_sum(x, y, s_los.astype(np.float32), fov, N_img)
    count = lu.bin2d_count(x, y, fov, N_img)
    mean_vz = np.zeros_like(vz_sum, dtype=np.float32)
    mask = count > 0
    mean_vz[mask] = vz_sum[mask] / count[mask]
    return mean_vz


def make_galaxy_los_disp_xy(x, y, s_los, fov, N_img):
    """Per-pixel LOS dispersion; empty and 1-galaxy pixels are 0."""
    vz = s_los.astype(np.float32)
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


def build_density_cube(components, fov, cache_path, h_floor_mpc):
    """
    Adaptive-SPH density on the finest grid, max(CFG_GRID.resolutions)^3 over
    [-fov, fov]^3 (Msun/Mpc^3, zyx), plus the mass falling outside it.

    Every particle of the cutout enters, inside the cube or not, so kernels
    near the faces see their full neighbourhood. h_j comes from the
    per-cluster cache: it is a function of inter-particle distances alone,
    which the projection's rotation does not change.
    """
    pos = np.concatenate([p for p, _ in components.values()], axis=0)
    mass = np.concatenate([m for _, m in components.values()], axis=0)
    h = du.cached_smoothing_lengths(cache_path, pos, CFG_DENS.n_ngb, h_floor_mpc,
                                    workers=SPH_THREADS)
    rho, outside = du.sph_density_cube(pos, mass, h, fov, max(CFG_GRID.resolutions),
                                       sub=CFG_DENS.sub, n_threads=SPH_THREADS)
    return rho.astype(np.float32), outside


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
    """
    Continuous pixel coordinates on the N_img x N_img image grid, pixel-centre
    convention: pixel i spans [i - 0.5, i + 0.5), so floor(coord + 0.5) is the
    bin lu.bin2d_sum put the galaxy in. (The old x01 * (N_img - 1) drifted up
    to half a pixel from that binning.)
    """
    x01 = np.clip((x + fov) / (2.0 * fov), 0.0, 1.0)
    y01 = np.clip((y + fov) / (2.0 * fov), 0.0, 1.0)
    return np.stack([x01 * N_img - 0.5, y01 * N_img - 0.5], axis=-1).astype(np.float32)


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
            for k in ("x", "y", "z_s", "z_true", "e1_obs", "e2_obs")}


def lovoccs_background_template(z_l):
    """
    The slice of the empirical n(z) that survives LoVoCCS II's background cuts
    for a lens at z_l: a foreground cut at z_l + src_z_foreground_pad and a
    high-z cut at src_z_max.

    Applied to the template *before* drawing rather than to the drawn catalog,
    which is what keeps the surviving density at n_source_per_arcmin2. LoVoCCS's
    ~7 arcmin^-2 is itself quoted after these cuts (it falls from ~22 arcmin^-2
    with shape and photometry cuts alone), so cutting a 7 arcmin^-2 draw down
    afterwards would double-count the loss.

    _EMPIRICAL_Z_TEMPLATE is pre-sorted in main(), so this is two searchsorteds.
    """
    z_lo = z_l + CFG_LENS.src_z_foreground_pad
    # side='right' / 'left' make both bounds strict: z_lo < z_s < src_z_max.
    lo = np.searchsorted(_EMPIRICAL_Z_TEMPLATE, z_lo, side="right")
    hi = np.searchsorted(_EMPIRICAL_Z_TEMPLATE, CFG_LENS.src_z_max, side="left")
    return _EMPIRICAL_Z_TEMPLATE[lo:hi]


def bpz_true_redshifts(z_phot, rng):
    """
    True redshifts behind a set of BPZ point estimates.

    The empirical template is built from real LoVoCCS BPZ z_b values, so a drawn
    z is what an observer *measures*; the truth scatters about it. LoVoCCS III
    quotes sigma_z/(1+z) ~ 0.051 with a catastrophic-outlier rate eta ~ 5.3%,
    so the core gets a Gaussian of that width and the outliers are redrawn from
    the full n(z) -- including in front of the lens, which is the point. A
    source whose z_phot passes the background cut but whose z_true sits at or
    below z_lens is unlensed, and dilutes the shear exactly the way real
    contamination does.

    Direction, stated plainly because it is an approximation: 0.051 is the
    scatter of z_phot about z_spec, i.e. it describes p(z_phot | z_true), and
    inverting it properly needs the true n(z) as a prior. Applying it
    symmetrically as p(z_true | z_phot) is the usual mock-catalog shortcut and
    is what happens here.
    """
    n = z_phot.size
    z_true = z_phot + rng.normal(0.0, CFG_LENS.bpz_sigma_z * (1.0 + z_phot))

    outlier = rng.random(n) < CFG_LENS.bpz_outlier_rate
    n_out = int(np.count_nonzero(outlier))
    if n_out:
        # From the *whole* template, not the background-cut slice: a
        # catastrophic failure can land anywhere, foreground included.
        z_true[outlier] = lu.sample_empirical_redshifts(
            n_out, _EMPIRICAL_Z_TEMPLATE, rng=rng)

    return np.maximum(z_true, 0.0), outlier


def draw_shape_catalog(kappa_inf, gamma1_inf, gamma2_inf, sigma_crit_inf,
                       fov_mpc, z_l, rng):
    """
    Synthetic source catalog:
      - n_source = CFG_LENS.n_source_per_arcmin2 over the full rectangular FoV
      - per-galaxy z_phot bootstrapped from the empirical n(z) template,
        restricted to LoVoCCS's background selection for this lens redshift
      - a true redshift behind each z_phot (bpz_true_redshifts). The lensing
        uses z_true; the catalog reports z_phot, because that is all an observer
        has. Sources scattered in front of the lens are simply not sheared.
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

    # Expected number of sources over the rectangular FoV. The core is no longer
    # cut geometrically: the BCG's occultation disc removes it on physical
    # grounds, in build_one_sample.
    field_side_arcmin = 2.0 * fov_mpc * arcmin_per_mpc
    field_area_arcmin2 = field_side_arcmin ** 2

    n_mean = CFG_LENS.n_source_per_arcmin2 * field_area_arcmin2

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

    # Sources uniformly over the rectangular FoV.
    x_g = rng.uniform(-fov_mpc, fov_mpc, size=n_proposed)
    y_g = rng.uniform(-fov_mpc, fov_mpc, size=n_proposed)

    z_template = lovoccs_background_template(z_l)
    if z_template.size == 0:
        raise RuntimeError(
            f"no template sources survive the LoVoCCS background cuts at "
            f"z_l={z_l:.4f}: need {z_l + CFG_LENS.src_z_foreground_pad:.3f} "
            f"< z_s < {CFG_LENS.src_z_max}"
        )
    # What the observer measures. The LoVoCCS background cuts are applied to
    # this, on the template, so they neither bias the count nor peek at truth.
    z_phot = lu.sample_empirical_redshifts(n_proposed, z_template, rng=rng)

    # What actually lenses. Some of these fall in front of the cluster despite
    # z_phot passing the cut; that is the contamination real catalogs carry.
    if CFG_LENS.apply_photoz_error:
        z_true, _ = bpz_true_redshifts(z_phot, rng)
    else:
        z_true = z_phot

    # Per-source critical surface density Sigma_crit(z_l, z_true); foreground
    # sources get Sigma_crit -> +inf and thus feel no lensing.
    sigma_crit_src = lu.sigma_crit_msun_per_mpc2(
        z_l, z_true, H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0)

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

    x_g, y_g = x_g[keep_mag], y_g[keep_mag]
    z_phot, z_true = z_phot[keep_mag], z_true[keep_mag]
    k_eff = k_eff[keep_mag]; g1_eff = g1_eff[keep_mag]; g2_eff = g2_eff[keep_mag]

    # reduced shear; reject |g| >= 1 (interior of critical curve)
    g_complex = (g1_eff + 1j * g2_eff) / (1.0 - k_eff + 1e-30)
    keep_g = np.abs(g_complex) < 1.0
    x_g, y_g = x_g[keep_g], y_g[keep_g]
    z_phot, z_true = z_phot[keep_g], z_true[keep_g]
    g_complex = g_complex[keep_g]

    # Intrinsic ellipticities, drawn wide and truncated to the physical |e| < 1
    # so the catalog realizes CFG_LENS.sigma_e_realized per component.
    e_int = lu.draw_intrinsic_ellipticity(g_complex.shape[0], SIGMA_E_INPUT, rng=rng)
    e_obs = (e_int + g_complex) / (1.0 + np.conjugate(g_complex) * e_int)

    return {
        "x": x_g.astype(np.float32),
        "y": y_g.astype(np.float32),
        # z_s is the BPZ point estimate: the observable, and the only redshift
        # that reaches the stored catalog. z_true is carried alongside purely so
        # build_one_sample can report how much foreground contamination the
        # photo-z scatter let through; it is never written per source.
        "z_s": z_phot.astype(np.float32),
        "z_true": z_true.astype(np.float32),
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

    Returns (src_features (N,8) f32, src_cell_id (N,) i32, src_cell_ptr (S*S+1,) i64).
    Feature columns: e1, e2, ix, iy, z_s, z_lens, cell_frac, coverage.

    `coverage` is the fraction of the pointing's exposures covering that source,
    i.e. the survey weight map an observer would have alongside the catalog.
    """
    N = cat["x"].size
    if N == 0:
        return (np.zeros((0, 8), dtype=np.float32),
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
    coverage = cat["coverage"][order]

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
        coverage.astype(np.float32),
    ], axis=-1)

    return src_features, cell_id.astype(np.int32), cell_ptr


# ============================================================
# Mock X-ray mosaic (stage 2 of the X-ray pipeline)
# ============================================================

_XU = None
# L_X500c depends only on (sim, cluster), not on the viewing direction, so the
# ~2 s spherical-overdensity solve is cached rather than paid once per
# projection. Per worker process, so it holds n_projections entries at most.
_LX_CACHE = {}


def _xray_module():
    """
    Import xray_utils lazily: it pulls in yt, pyXSIM and SOXS, which is slow and
    pointless for a run with CFG_XRAY.enabled = False.
    """
    global _XU
    if _XU is None:
        import xray_utils
        xray_utils.quiet_logs()
        _XU = xray_utils
    return _XU


def check_photon_lists(jobs):
    """
    Fail before the build starts, not two hours in, if xray_photons.slurm has
    not covered every (sim, cluster, projection) the sample list needs.

    Also checks the redshift each list was *actually* generated at against
    config.sample_z_lens. The two stages agree only because they call the same
    keyed function; if that ever drifts -- a changed z_lens_seed, an edited
    z_lens range, a stale list left over from an earlier run -- the maps would
    silently be at the wrong distance, so it is worth the h5 open per list.
    """
    if not CFG_XRAY.enabled:
        print("X-ray maps disabled (CFG_XRAY.enabled = False)")
        return

    xu = _xray_module()
    needed, missing, mismatched = set(), set(), []
    for _, sim, cluster_idx, proj_index, _, _ in jobs:
        key = (sim, cluster_idx, proj_index)
        if key in needed:
            continue
        needed.add(key)
        prefix = xu.photon_prefix(sim, cluster_idx, proj_index)
        if not xu.photon_list_is_complete(prefix):
            missing.add(key)
            continue
        z_want = sample_z_lens(sim, cluster_idx, proj_index)
        z_have = xu.photon_list_redshift(prefix)
        if not np.isfinite(z_have) or abs(z_have - z_want) > 1e-6:
            mismatched.append((sim, cluster_idx, proj_index, z_want, z_have))

    if missing:
        examples = "\n  ".join(
            f"{s} GrNm_{c:03d} p{p} (z={sample_z_lens(s, c, p):.5f})"
            for s, c, p in sorted(missing)[:5])
        raise FileNotFoundError(
            f"{len(missing)} of {len(needed)} photon lists are missing or "
            f"incomplete in {CFG_XRAY.photon_dir}, e.g.:\n  {examples}\n"
            f"Run xray_photons.slurm first, or set CFG_XRAY.enabled = False."
        )
    if mismatched:
        examples = "\n  ".join(
            f"{s} GrNm_{c:03d} p{p}: want z={zw:.6f}, file has z={zh:.6f}"
            for s, c, p, zw, zh in mismatched[:5])
        raise RuntimeError(
            f"{len(mismatched)} of {len(needed)} photon lists were generated at "
            f"a different redshift than config.sample_z_lens now returns, "
            f"e.g.:\n  {examples}\n"
            f"CFG_LENS.z_lens_seed / z_lens_min / z_lens_max or "
            f"CFG_DATA.n_projections must have changed since xray_photons.slurm "
            f"ran. Regenerate the lists (--overwrite) or restore the old values."
        )
    n_pt = CFG_XRAY.n_side ** 2
    print(f"X-ray: {len(needed)} photon lists present in {CFG_XRAY.photon_dir}; "
          f"{n_pt} {CFG_XRAY.instrument} pointing{'s' if n_pt > 1 else ''}, "
          f"exposure ~lognormal(median {CFG_XRAY.t_exp_median_ks} ks, "
          f"sigma_ln {CFG_XRAY.t_exp_sigma_ln}) clipped to "
          f"[{CFG_XRAY.t_exp_min_ks}, {CFG_XRAY.t_exp_max_ks}] ks, "
          f"{CFG_XRAY.band_emin_kev}-{CFG_XRAY.band_emax_kev} keV, stored "
          f"{CFG_XRAY.grid_size}^2 over {CFG_XRAY.fov_arcmin}'")


def build_xray_maps(sim, cluster_idx, proj_index, proj_vec, z_lens, h_sim,
                    seed, t_exp_ks):
    """
    Project this sample's photon list along its viewing direction and observe it
    for t_exp_ks. By far the most expensive thing in build_one_sample.

    make_photons already ran in xray_photons.py, once per (sim, cluster,
    projection) at exactly this sample's z_lens; here we do the parts that
    depend on the viewing direction and the depth.
    """
    if not CFG_XRAY.enabled:
        return None

    xu = _xray_module()
    photon_prefix = xu.photon_prefix(sim, cluster_idx, proj_index)
    if not os.path.exists(photon_prefix + ".h5"):
        raise FileNotFoundError(
            f"no photon list for {sim} GrNm_{cluster_idx:03d} p{proj_index} "
            f"(z={z_lens:.5f}) at {photon_prefix}.h5. Run xray_photons.slurm "
            f"first, or set CFG_XRAY.enabled = False."
        )

    # Relative to the worker's CWD (CFG_XRAY.work_dir, set in _init_worker), so
    # SOXS's SIMPUT/photon-list cross-references resolve the way they do in the
    # notebook. The seed makes it unique across samples.
    work_prefix = f"s_{sim}_{cluster_idx:03d}_{seed}"
    try:
        products = xu.project_and_observe(photon_prefix, work_prefix, proj_vec,
                                          t_exp_ks, seed=seed)
        # The photon list is at this sample's own z_lens, so the angular scale
        # the maps are stored with is the same one the lensing geometry uses.
        maps = xu.build_sample_maps(products, z_lens, h_sim)
    finally:
        xu.clean_intermediates(work_prefix)     # never touches the photon list
    return standardize_xray_maps(maps)


def standardize_xray_maps(maps):
    """
    Put xray_mosaic and xray_counts into the form the model consumes, in place:
    a validity mask, and log10 z-scores on the pixels that mask keeps.

    Only the pixels where the detector looked (xray_expmap > 0) and a photon
    landed (xray_counts > CFG_XRAY.min_counts) carry information; everywhere
    else log10 is either undefined or would be reporting reprojection leakage
    (see CFG_XRAY.min_counts). Those pixels become

        (log10(x) - <log10 x>) / sigma(log10 x)

    with the constants fixed in CFG_XRAY, and the rest are set to 0 -- a filler
    the mask tells the model to ignore, not a measurement of an empty pixel.

    The raw maps are not kept alongside these: xray_counts inverts exactly as
    10**(z * std + mean), and xray_mosaic is that divided by
    xray_expmap * t_exp_s * arcsec_per_pix^2, so nothing is lost.

    xray_expmap and xray_ideal are left in physical units.
    """
    counts = maps["xray_counts"].astype(np.float64)
    expmap = maps["xray_expmap"].astype(np.float64)
    mosaic = maps["xray_mosaic"].astype(np.float64)

    # 1.0 = usable, matching the polarity of the `mask` dataset on the member
    # point cloud. Both criteria are needed to make the log10s below finite.
    valid = (expmap > 0.0) & (counts > CFG_XRAY.min_counts)

    def _log10_zscore(x, mean, std):
        out = np.zeros(x.shape, dtype=np.float32)
        if np.any(valid):
            out[valid] = ((np.log10(x[valid]) - mean) / std).astype(np.float32)
        return out

    maps["xray_mosaic"] = _log10_zscore(
        mosaic, CFG_XRAY.mosaic_log10_mean, CFG_XRAY.mosaic_log10_std)
    maps["xray_counts"] = _log10_zscore(
        counts, CFG_XRAY.counts_log10_mean, CFG_XRAY.counts_log10_std)
    maps["xray_mask"] = valid.astype(np.float32)
    return maps


# ============================================================
# Sample builder (one viewing frame per sample)
# ============================================================

def build_one_sample(args) -> Dict:
    npz_path, sim, cluster_idx, proj_index, proj_vec, seed = args
    rng = np.random.default_rng(seed)

    # Per-sample lens redshift. Not drawn from `rng`: xray_photons.py had to
    # know it, in a separate slurm job, to generate the photon list at the right
    # redshift, so it comes from the identity-keyed config.sample_z_lens that
    # both stages share.
    z_lens = sample_z_lens(sim, cluster_idx, proj_index)

    data = np.load(npz_path)
    h_sim = float(data["h"])
    a_scale = float(data["a"])
    boxsize = BOXSIZE_MPC_OVER_H
    fov = CFG_GRID.fov_mpc

    # -------------------------------
    # Cluster-member galaxies -> viewing frame
    # -------------------------------
    if _MAG_CALIBRATION is None:
        raise RuntimeError(
            "build_one_sample: worker has no g-band calibration "
            "(Pool initializer was not run)."
        )
    mstar_cal, mag_g_cal = _MAG_CALIBRATION
    sub_mstar = data["sub_massTotal"][:, 4]        # Msun, already h-corrected

    # BCG = most massive stellar subhalo within bcg_search_radius_r200 * R200 of
    # the FoF centre of potential, and the cluster is centred on it: the BCG is
    # observable, whereas the centre of potential is a halo-finder construct.
    # The aperture keeps the argmax off a neighbouring cluster's BCG, which the
    # 5 R200 cutout can contain (a handful of clusters, at 2.7-4.3 R200).
    d_cop = data["sub_pos"] - data["CoP"]
    d_cop = (d_cop + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    near_center = (np.linalg.norm(d_cop, axis=1)
                   <= CFG_DATA.bcg_search_radius_r200 * float(data["R200"]))
    if not np.any(near_center):
        near_center = np.ones(sub_mstar.shape, dtype=bool)
    near_inds = np.flatnonzero(near_center)
    bcg_idx = int(near_inds[np.argmax(sub_mstar[near_inds])])
    center = data["sub_pos"][bcg_idx]

    # Luminous-galaxy cut: P(M_g <= mag_cut_g | Mstar) calibrated on TNG300,
    # which BAHAMAS cannot do directly for lack of photometry.
    bright = mag_g_proxy_mask(
        mstar_cal=mstar_cal,
        mag_g_cal=mag_g_cal,
        mstar_target=sub_mstar,
        mag_cut=CFG_DATA.mag_cut_g,
        method="stochastic",
        seed=CFG_DATA.galaxy_cut_seed,
    )
    bright[bcg_idx] = True     # the BCG defines the frame and the redshift zero-point

    difpos = data["sub_pos"][bright] - center
    coords = (difpos + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    pos = (coords / (h_sim * a_scale)) + EPS  # Mpc

    # Velocities in the BCG rest frame, to match the BCG-centred redshift zero-point.
    vel = data["sub_vel"][bright] - data["sub_vel"][bcg_idx]

    ro_pos, ro_vel = TNG_DA.rotate_to_viewing_frame(pos, vel, proj_vec)
    x, y, z = (ro_pos[:, i].astype(np.float32) for i in range(3))
    vx, vy, vz = (ro_vel[:, i].astype(np.float32) for i in range(3))

    # Members in redshift space about the BCG at z_lens; the observed redshift
    # replaces v_z as the model-facing LOS coordinate.
    z_gal = members_to_redshift(z.astype(np.float64), vz.astype(np.float64), z_lens)
    z_gal_s = ((z_gal - CFG_SCALE.z_mean) / CFG_SCALE.z_std).astype(np.float32)

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
        dif = pos_raw - center
        cc = (dif + 0.5 * boxsize) % boxsize - 0.5 * boxsize
        pos_mpc = (cc / (h_sim * a_scale)) + EPS
        # The particle masses are already Msun -- h is divided out in the cutout
        # files. Verified three ways rather than taken from a comment: dm_mass
        # is a single value 5.5065e9, which is exactly BAHAMAS's published
        # 3.85e9 h^-1 Msun at h = 0.7; gas_mass's floor 1.0942e9 is the
        # published 7.66e8 h^-1 Msun; and summing every species inside R200
        # gives 0.9997 x M200 (median over all 100 CDMb clusters, range
        # 0.996-1.000), which would be 1.428 x M200 with a 1/h applied.
        # An earlier 1/h here inflated Sigma -> kappa -> gamma by 43%.
        mass_msun = mass_raw                                         # already Msun
        rot, _ = TNG_DA.rotate_to_viewing_frame(pos_mpc, np.zeros_like(pos_mpc), proj_vec)
        return rot.astype(np.float32), mass_msun.astype(np.float32)

    components = {
        "dm":   prep_component("dm_pos",   "dm_mass"),
        "gas":  prep_component("gas_pos",  "gas_mass"),
        "star": prep_component("star_pos", "star_mass"),
        "bh":   prep_component("bh_pos",   "bh_mass"),
    }

    # -------------------------------
    # Dynamics conditioning images (mass_xy replaced by the source catalog),
    # one (3, N, N) stack per resolution
    # -------------------------------
    images = {}
    for N_img in CFG_GRID.resolutions:
        gal_xy = make_galaxy_map_xy(x, y, fov, N_img)
        # Mean channel: per-pixel mean of the standardized member redshift
        # (empty pixels stay at 0). Dispersion channel: per-pixel redshift
        # dispersion on its own scale, which is ~z_std/8 and would otherwise be
        # a flat image.
        gal_z_xy = make_galaxy_los_mean_xy(x, y, z_gal_s, fov, N_img)
        gal_z_disp_xy = (make_galaxy_los_disp_xy(x, y, z_gal.astype(np.float32),
                                                fov, N_img) / CFG_SCALE.z_disp_std)
        images[N_img] = np.stack([gal_xy, gal_z_xy, gal_z_disp_xy],
                                 axis=0).astype(np.float32)

    # -------------------------------
    # 3D target cube + global targets
    # -------------------------------
    # Only the finest cube is built here; main() block-averages it to the
    # other resolutions. The global targets are physical quantities, so they
    # come from this most accurate cube and are shared by every N.
    h_cache = os.path.join(CFG_DENS.cache_dir, sim,
                           f"GrNm_{cluster_idx:03d}_ngb{CFG_DENS.n_ngb}.npz")
    density_cube, mass_outside = build_density_cube(components, fov, h_cache,
                                                    sph_h_floor_mpc(sim))
    cube_mass_msun = rho_cube_to_mass_msun(density_cube, fov)
    cube_mass_log10_msun = np.log10(max(cube_mass_msun, 1e-30))
    axis_lengths_mpc = np.array(rho_cube_to_axis_lengths_mpc(density_cube, fov),
                                dtype=np.float32)

    # -------------------------------
    # Cluster-member point cloud (padded)
    # -------------------------------
    x_s = (x - CFG_SCALE.pos_mean) / CFG_SCALE.pos_std
    y_s = (y - CFG_SCALE.pos_mean) / CFG_SCALE.pos_std
    z_sc = (z - CFG_SCALE.pos_mean) / CFG_SCALE.pos_std
    vx_s = (vx - CFG_SCALE.vel_mean) / CFG_SCALE.vel_std
    vy_s = (vy - CFG_SCALE.vel_mean) / CFG_SCALE.vel_std

    N_gal = x.shape[0]
    n_feat = np.full((N_gal,), float(N_gal), dtype=np.float32)
    gal_features = np.stack([x_s, y_s, z_gal_s, n_feat], axis=-1).astype(np.float32)
    gal_targets = np.stack([z_sc, vx_s, vy_s], axis=-1).astype(np.float32)

    max_nodes = CFG_DATA.max_nodes
    n_keep = min(N_gal, max_nodes)
    feat_pad = np.zeros((max_nodes, 4), dtype=np.float32)
    targ_pad = np.zeros((max_nodes, 3), dtype=np.float32)
    mask = np.zeros((max_nodes,), dtype=np.float32)
    feat_pad[:n_keep] = gal_features[:n_keep]
    targ_pad[:n_keep] = gal_targets[:n_keep]
    mask[:n_keep] = 1.0
    pix_pad = {}
    for N_img in CFG_GRID.resolutions:
        pix_pad[N_img] = np.zeros((max_nodes, 2), dtype=np.float32)
        pix_pad[N_img][:n_keep] = xy_to_pixel_coords(x, y, fov, N_img)[:n_keep]

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

    # -------------------------------
    # Survey masks: member occultation, then the dithered pointing footprint
    # -------------------------------
    # Radii in Mpc; the BCG's row in the post-cut member arrays is the number of
    # selected members ahead of it.
    bcg_row = int(np.count_nonzero(bright[:bcg_idx]))
    mem_radii = np.full(N_gal, CFG_LENS.member_mask_radius_kpc / 1e3)
    mem_radii[bcg_row] = CFG_LENS.bcg_mask_radius_kpc / 1e3

    keep_occult = lu.occultation_keep_mask(cat["x"], cat["y"], x, y, mem_radii)

    pointing = lu.draw_pointing(
        rng, z_lens,
        side_arcmin_range=CFG_LENS.pointing_side_arcmin,
        dither_amp_arcmin_range=CFG_LENS.dither_amp_arcmin,
        n_exposures=CFG_LENS.n_exposures,
        scale_scatter=CFG_LENS.pointing_scale_scatter,
        center_offset_frac=CFG_LENS.pointing_center_offset_frac,
        H0=CFG_COSMO.H0, Om0=CFG_COSMO.Om0,
    )
    coverage = lu.pointing_coverage_fraction(cat["x"], cat["y"], pointing)
    keep_pointing = rng.random(coverage.shape) < coverage ** CFG_LENS.coverage_exponent

    keep = keep_occult & keep_pointing
    n_drawn = int(cat["x"].size)
    cat = {k: v[keep] for k, v in cat.items()}
    cat["coverage"] = coverage[keep].astype(np.float32)

    src_coverage_map = lu.coverage_map(pointing, fov, S)
    keep_fracs = (
        float(keep_occult.mean()) if n_drawn else 1.0,
        float(keep_pointing.mean()) if n_drawn else 1.0,
        float(keep.mean()) if n_drawn else 1.0,
    )

    # How much of the surviving catalog is really in front of the lens despite
    # passing the z_phot cut. Reported as a scalar, not per source: z_true never
    # reaches the stored catalog, so a model cannot learn its way around the
    # dilution, but the systematic can still be measured after the fact.
    n_kept = int(cat["x"].size)
    src_true_foreground_frac = (
        float(np.mean(cat["z_true"] <= z_lens)) if n_kept else 0.0)
    src_z_true_mean = float(np.mean(cat["z_true"])) if n_kept else float("nan")
    cat.pop("z_true")

    src_features, src_cell_id, src_cell_ptr = process_source_catalog(
        cat, z_lens, fov, S)

    # -------------------------------
    # Mock Chandra ACIS-I observation, in this same viewing frame
    # -------------------------------
    # Archival depth varies cluster to cluster, so the exposure is drawn per
    # sample. Drawn from `rng` after everything above, so adding it does not
    # move any earlier draw (z_lens in particular, which check_photon_lists
    # reproduces from the seed).
    t_exp_ks = CFG_XRAY.draw_exposure_ks(rng) if CFG_XRAY.enabled else 0.0
    xray = build_xray_maps(sim, cluster_idx, proj_index, proj_vec, z_lens,
                           h_sim, seed, t_exp_ks)

    # LoVoCCS's X-ray selection is L_X500c(0.1-2.4 keV) > 1e44 erg/s. Stored so
    # that selection can be inspected after the fact; nothing is cut on it.
    if CFG_XRAY.enabled:
        cache_key = (sim, cluster_idx)
        if cache_key not in _LX_CACHE:
            xu = _xray_module()
            lx, r5, _ = xu.lx500c_erg_s(data, center)
            _LX_CACHE[cache_key] = (lx, r5)
        lx500c, r500c = _LX_CACHE[cache_key]
    else:
        lx500c, r500c = float("nan"), float("nan")

    return dict(
        sim=str(sim),
        cluster_idx=np.int32(cluster_idx),
        proj_index=np.int32(proj_index),
        halo_mass=np.float32(data["Mfof"]),
        proj_vec=np.asarray(proj_vec, dtype=np.float32),
        # float64: z_lens is the contract between this build and
        # xray_photons.py, so it has to round-trip against sample_z_lens exactly
        # rather than to float32's ~7 digits.
        z_lens=np.float64(z_lens),

        images=images,                     # {N: (3, N, N)}

        gal_features=feat_pad,
        gal_targets=targ_pad,
        gal_pixel_coords=pix_pad,          # {N: (max_nodes, 2)}
        mask=mask,
        n_gal=np.int32(N_gal),

        src_features=src_features,
        src_cell_id=src_cell_id,
        src_cell_ptr=src_cell_ptr,
        n_src=np.int32(src_features.shape[0]),
        n_src_drawn=np.int32(n_drawn),
        src_coverage=src_coverage_map,
        pointing=pointing,
        keep_fracs=keep_fracs,
        src_true_foreground_frac=np.float32(src_true_foreground_frac),
        src_z_true_mean=np.float32(src_z_true_mean),

        raw_density_cube=density_cube,     # finest resolution only
        sph_mass_outside_msun=np.float64(mass_outside),
        cube_mass_log10_msun=np.float32(cube_mass_log10_msun),
        axis_lengths_mpc=axis_lengths_mpc,

        xray=xray,
        lx500c_erg_s=np.float64(lx500c),
        r500c_mpc_over_h=np.float64(r500c),
    )


def build_cluster_samples(cluster_jobs):
    """
    Every projection of one (sim, cluster), on one worker. Grouped so the
    first projection writes the cluster's h_j cache and the rest read it back,
    instead of all projections running the kNN at once on different workers.
    """
    return [build_one_sample(job) for job in cluster_jobs]


class _LogStats:
    """Running count / sum / sum of squares of log10(cube) over positive voxels."""

    def __init__(self):
        self.n, self.s, self.s2 = 0, 0.0, 0.0

    def add(self, cube):
        v = cube[cube > 0.0]
        if v.size:
            lv = np.log10(v.astype(np.float64))
            self.n += lv.size
            self.s += float(np.sum(lv))
            self.s2 += float(np.sum(lv * lv))

    def mean_std(self):
        if self.n == 0:
            raise RuntimeError("no positive-valued cube voxels found")
        mean = self.s / self.n
        return mean, float(np.sqrt(max(self.s2 / self.n - mean * mean, 0.0)) + 1e-6)


# ============================================================
# Writing
# ============================================================

def write_static_attrs(f: h5py.File):
    f.attrs["map_fov_mpc"] = CFG_GRID.fov_mpc
    # Every resolution-dependent product is stored once per N here, suffixed
    # _N: images_N (3,N,N), gal_pixel_coords_N, density_cube_N (N,N,N), and the
    # file attrs cube_log10_{mean,std}_N.
    f.attrs["resolutions"] = np.array(CFG_GRID.resolutions, dtype=np.int64)
    f.attrs["vz_max_kms"] = CFG_GRID.vz_max_kms
    f.attrs["floor_value"] = CFG_GRID.floor_value
    f.attrs["max_nodes"] = CFG_DATA.max_nodes

    f.attrs["pos_mean"] = CFG_SCALE.pos_mean
    f.attrs["pos_std"] = CFG_SCALE.pos_std
    f.attrs["vel_mean"] = CFG_SCALE.vel_mean
    f.attrs["vel_std"] = CFG_SCALE.vel_std
    f.attrs["z_mean"] = CFG_SCALE.z_mean
    f.attrs["z_std"] = CFG_SCALE.z_std
    f.attrs["z_disp_std"] = CFG_SCALE.z_disp_std
    f.attrs["h"] = CFG_COSMO.h_sim
    f.attrs["Om0_sim"] = CFG_COSMO.Om0_sim

    # Member selection and the BCG-centred, redshift-space LOS coordinate.
    f.attrs["mag_cut_g"] = CFG_DATA.mag_cut_g
    f.attrs["galaxy_cut_seed"] = CFG_DATA.galaxy_cut_seed
    f.attrs["galaxy_cut_calibration"] = ",".join(CFG_DATA.tng_calibration_sims)
    f.attrs["bcg_search_radius_r200"] = CFG_DATA.bcg_search_radius_r200
    f.attrs["cluster_center"] = "BCG (most massive stellar subhalo within R200 of CoP)"
    f.attrs["los_feature_definition"] = (
        "(z_gal - z_mean)/z_std, where z_gal is the observed member redshift: "
        "BCG at z_lens, other members placed in redshift space by peculiar "
        "velocity + H(z_lens) Hubble flow in the BAHAMAS cosmology. "
        "gal_z_disp_xy is the per-pixel dispersion of z_gal over z_disp_std."
    )

    # Weak-lensing source-catalog metadata (model-facing).
    f.attrs["shape_grid_size"] = CFG_LENS.shape_grid_size
    f.attrs["shape_fov_mpc"] = CFG_GRID.fov_mpc
    f.attrs["z_lens_min"] = CFG_LENS.z_lens_min
    f.attrs["z_lens_max"] = CFG_LENS.z_lens_max
    f.attrs["z_lens_seed"] = CFG_LENS.z_lens_seed
    f.attrs["z_lens_note"] = (
        "z_lens = config.sample_z_lens(simulation, cluster_index, "
        "projection_index), keyed on the sample's identity rather than a "
        "sequential rng stream so xray_photons.py and this build agree on it "
        "across separate slurm jobs")
    f.attrs["n_source_per_arcmin2"] = CFG_LENS.n_source_per_arcmin2
    f.attrs["sigma_e_realized"] = CFG_LENS.sigma_e_realized
    # LoVoCCS background selection and BPZ error model.
    f.attrs["src_z_foreground_pad"] = CFG_LENS.src_z_foreground_pad
    f.attrs["src_z_max"] = CFG_LENS.src_z_max
    f.attrs["src_selection"] = (
        "LoVoCCS II: z_lens + src_z_foreground_pad < z_phot < src_z_max, "
        "applied to the empirical n(z) before drawing so the surviving density "
        "is still n_source_per_arcmin2")
    f.attrs["bpz_sigma_z"] = CFG_LENS.bpz_sigma_z
    f.attrs["bpz_outlier_rate"] = CFG_LENS.bpz_outlier_rate
    f.attrs["apply_photoz_error"] = bool(CFG_LENS.apply_photoz_error)
    f.attrs["src_z_s_definition"] = (
        "BPZ point estimate (the observable). Lensing was computed with a true "
        "redshift scattered about it: Gaussian sigma_z = bpz_sigma_z*(1+z) for "
        "the core and a bpz_outlier_rate fraction redrawn from the full n(z). "
        "The true redshifts are not stored per source; the resulting dilution "
        "is summarised per sample as src_true_foreground_frac.")
    f.attrs["sigma_e_gaussian_input"] = SIGMA_E_INPUT

    # Survey masks.
    f.attrs["bcg_mask_radius_kpc"] = CFG_LENS.bcg_mask_radius_kpc
    f.attrs["member_mask_radius_kpc"] = CFG_LENS.member_mask_radius_kpc
    f.attrs["pointing_side_arcmin"] = np.array(CFG_LENS.pointing_side_arcmin)
    f.attrs["pointing_scale_scatter"] = CFG_LENS.pointing_scale_scatter
    f.attrs["pointing_center_offset_frac"] = CFG_LENS.pointing_center_offset_frac
    f.attrs["n_exposures"] = CFG_LENS.n_exposures
    f.attrs["dither_amp_arcmin"] = np.array(CFG_LENS.dither_amp_arcmin)
    f.attrs["coverage_exponent"] = CFG_LENS.coverage_exponent
    f.attrs["src_coverage_units"] = (
        "fraction of the pointing's exposures covering the cell; "
        "0 = never observed, 1 = every exposure")
    f.attrs["empirical_template"] = CFG_LENS.empirical_template
    f.attrs["empirical_redshift_npz_path"] = CFG_LENS.empirical_redshift_npz_path
    f.attrs["src_feature_columns"] = np.array(
        [b"e1", b"e2", b"ix", b"iy", b"z_s", b"z_lens", b"cell_frac", b"coverage"],
        dtype="S12")

    f.attrs["image_channels"] = np.array(
        [b"gal_xy", b"gal_z_xy", b"gal_z_disp_xy"], dtype="S20")
    f.attrs["density_cube_order"] = "zyx"
    f.attrs["density_estimator"] = (
        f"adaptive SPH: rho(x) = sum_j m_j W(|x - x_j|, h_j), Gadget cubic "
        f"spline, h_j = distance to the {CFG_DENS.n_ngb}th nearest neighbour "
        f"over all species of the cutout, floored at density_h_floor_mpc (per simulation). Each "
        f"voxel holds the voxel-averaged density (Msun/Mpc^3, before "
        f"standardization), with mass conserved per particle. Deposited once at "
        f"{max(CFG_GRID.resolutions)}^3; the coarser density_cube_N are exact "
        f"block averages of it, i.e. the same field over larger voxels.")
    f.attrs["density_n_ngb"] = CFG_DENS.n_ngb
    f.attrs["density_h_floor_mpc"] = ";".join(
        f"{s}:{sph_h_floor_mpc(s):.5f}" for s in CFG_DATA.simulations)
    f.attrs["density_sub_samples"] = CFG_DENS.sub
    f.attrs["globals_target_resolution"] = max(CFG_GRID.resolutions)
    f.attrs["gal_feature_columns"] = np.array([b"x", b"y", b"z_gal", b"Ngal"], dtype="S8")
    f.attrs["gal_target_columns"] = np.array([b"z", b"vx", b"vy"], dtype="S8")
    f.attrs["gal_pixel_coords_columns"] = np.array([b"x_pix", b"y_pix"], dtype="S8")
    f.attrs["gal_pixel_coords_convention"] = (
        "continuous coordinates on the N x N grid of images_N, pixel-centre "
        "convention: pixel i spans [i - 0.5, i + 0.5), so floor(coord + 0.5) "
        "is the image bin the galaxy was counted in")
    f.attrs["globals_target_columns"] = np.array(
        [b"mass_log10_msun", b"axis_a_mpc", b"axis_b_mpc", b"axis_c_mpc"], dtype="S20")

    # Mock X-ray mosaics: same viewing frame as the images above, but on their
    # own fixed *angular* grid, so xray_mpc_per_pix is a per-sample attribute.
    if CFG_XRAY.enabled:
        f.attrs["xray_grid_size"] = CFG_XRAY.grid_size
        f.attrs["xray_fov_arcmin"] = CFG_XRAY.fov_arcmin
        f.attrs["xray_instrument"] = CFG_XRAY.instrument
        f.attrs["xray_n_side"] = CFG_XRAY.n_side
        f.attrs["xray_n_pointings"] = CFG_XRAY.n_side ** 2
        f.attrs["xray_step_arcmin"] = CFG_XRAY.step_arcmin
        # Exposure is per sample now (attr xray_t_exp_ks on each group); these
        # are the parameters of the distribution it is drawn from.
        f.attrs["xray_t_exp_median_ks"] = CFG_XRAY.t_exp_median_ks
        f.attrs["xray_t_exp_sigma_ln"] = CFG_XRAY.t_exp_sigma_ln
        f.attrs["xray_t_exp_range_ks"] = np.array(
            [CFG_XRAY.t_exp_min_ks, CFG_XRAY.t_exp_max_ks], dtype=np.float64)
        f.attrs["xray_exposure_note"] = (
            "LoVoCCS X-ray coverage is archival Chandra/XMM, so depth varies "
            "cluster to cluster; each sample draws its own exposure and stores "
            "the realised value as the group attr xray_t_exp_ks")
        f.attrs["xray_band_kev"] = np.array(
            [CFG_XRAY.band_emin_kev, CFG_XRAY.band_emax_kev], dtype=np.float64)
        f.attrs["xray_redshift_note"] = (
            "one photon list per sample, generated at that sample's exact "
            "z_lens (config.sample_z_lens), so the group attrs xray_redshift "
            "and z_lens are equal by construction")
        f.attrs["xray_nH_gal"] = CFG_XRAY.nH_gal
        f.attrs["xray_Z_met"] = CFG_XRAY.Z_met
        f.attrs["xray_maps"] = np.array(
            [b"xray_mosaic", b"xray_counts", b"xray_expmap", b"xray_ideal",
             b"xray_mask"], dtype="S16")
        # xray_mosaic and xray_counts are stored model-ready: log10, z-scored
        # with the constants below, and zeroed wherever xray_mask is 0.
        f.attrs["xray_min_counts"] = CFG_XRAY.min_counts
        f.attrs["xray_mosaic_log10_mean"] = CFG_XRAY.mosaic_log10_mean
        f.attrs["xray_mosaic_log10_std"] = CFG_XRAY.mosaic_log10_std
        f.attrs["xray_counts_log10_mean"] = CFG_XRAY.counts_log10_mean
        f.attrs["xray_counts_log10_std"] = CFG_XRAY.counts_log10_std
        f.attrs["xray_mask_definition"] = (
            "1.0 where the pixel is usable, 0.0 where it is not -- same "
            "polarity as the `mask` dataset on the member point cloud. Usable "
            "means xray_expmap > 0 (the detector looked there) and raw counts "
            "> xray_min_counts (a photon landed, and the value is not the "
            "sub-count reprojection leakage soxs.make_mosaic_image leaves over "
            "~10% of the field). Only these pixels should enter the model's "
            "embeddings; xray_mosaic and xray_counts are 0 elsewhere, which is "
            "filler rather than a measurement.")
        f.attrs["xray_standardization"] = (
            "xray_mosaic and xray_counts hold (log10(raw) - mean) / std over "
            "the valid pixels, using xray_{mosaic,counts}_log10_{mean,std}. "
            "Raw counts come back as 10**(xray_counts * std + mean); raw "
            "surface brightness likewise, and equals those counts divided by "
            "xray_expmap * (1e3 * xray_t_exp_ks) * xray_arcsec_per_pix**2. The "
            "raw maps are therefore not stored a second time.")
        f.attrs["xray_mosaic_units"] = (
            "dimensionless, log10 z-scored (see xray_standardization); the "
            "underlying map is counts s^-1 cm^-2 arcsec^-2, exposure corrected")
        f.attrs["xray_counts_units"] = (
            "dimensionless, log10 z-scored (see xray_standardization); the "
            "underlying map is counts per stored pixel")
        f.attrs["xray_ideal_units"] = (
            "counts s^-1 cm^-2 arcsec^-2, no instrument, PSF or background "
            "(raw, not standardized)")
        f.attrs["xray_expmap_units"] = (
            "cm^2 (SOXS-normalised, i.e. exposure map / exposure time), "
            "mean over the stored pixel (raw, not standardized)")

    # Intrinsic X-ray luminosity: a diagnostic against LoVoCCS's selection,
    # never a cut on this dataset.
    f.attrs["lx500c_band_kev"] = np.array(CFG_XRAY.lx_band_kev, dtype=np.float64)
    f.attrs["lx500c_units"] = "erg s^-1, rest frame, inside R500c"
    f.attrs["lx500c_note"] = (
        "LoVoCCS selects L_X500c(0.1-2.4 keV) > 1e44 erg/s. This sample is "
        "mass-selected, so lx500c_erg_s is recorded per sample for comparison "
        "only; no simulated cluster is removed on it.")
    f.attrs["lx500c_method"] = (
        "sum of n_e n_H Lambda(T,Z) V over gas with T > T_min_K and "
        "n_H < nH_max inside R500c, with Lambda the band-integrated APEC "
        "energy cooling function (APEC v3.1.3 via soxs.spectra.ApecGenerator, "
        "abund_table angr). R500c solved from the particles themselves as the "
        "largest radius enclosing a mean density >= 500 rho_crit(z=0).")


def write_sample_streaming(f: h5py.File, sample_id: int, sample: Dict):
    """Write everything that does not depend on train-set statistics."""
    grp = f.create_group(f"{sample_id:06d}")
    grp.attrs["id"] = int(sample_id)
    grp.attrs["simulation"] = sample["sim"]
    grp.attrs["cluster_index"] = int(sample["cluster_idx"])
    # Identifies the sample to config.sample_z_lens and to the photon list it
    # was observed from; samples land in the file in completion order, so this
    # is the only way back to (simulation, cluster, projection).
    grp.attrs["projection_index"] = int(sample["proj_index"])
    grp.attrs["cluster_mass"] = float(sample["halo_mass"])
    grp.attrs["n_galaxies"] = int(sample["n_gal"])
    grp.attrs["n_sources"] = int(sample["n_src"])
    grp.attrs["z_lens"] = float(sample["z_lens"])

    grp.attrs["sph_mass_outside_msun"] = float(sample["sph_mass_outside_msun"])

    grp.create_dataset("projection_vector", data=sample["proj_vec"])
    for N in CFG_GRID.resolutions:
        grp.create_dataset(f"images_{N}", data=sample["images"][N],
                           compression="gzip")                                 # (3,N,N)
        grp.create_dataset(f"gal_pixel_coords_{N}", data=sample["gal_pixel_coords"][N],
                           compression="gzip")
    grp.create_dataset("gal_features", data=sample["gal_features"], compression="gzip")
    grp.create_dataset("gal_targets", data=sample["gal_targets"], compression="gzip")
    grp.create_dataset("mask", data=sample["mask"], compression="gzip")

    grp.create_dataset("src_features", data=sample["src_features"], compression="gzip")
    grp.create_dataset("src_cell_id", data=sample["src_cell_id"], compression="gzip")
    grp.create_dataset("src_cell_ptr", data=sample["src_cell_ptr"], compression="gzip")

    # Survey coverage: which parts of the field were observed, and how deeply.
    # Without it, an empty cell is ambiguous between 'no sources there' and
    # 'never pointed there'.
    grp.create_dataset("src_coverage", data=sample["src_coverage"], compression="gzip")
    pointing = sample["pointing"]
    grp.create_dataset("dither_offsets_mpc", data=pointing["offsets_mpc"].astype(np.float32))
    grp.attrs["pointing_center_mpc"] = pointing["center_mpc"].astype(np.float32)
    grp.attrs["pointing_sides_mpc"] = pointing["sides_mpc"].astype(np.float32)
    grp.attrs["pointing_angle_deg"] = float(np.degrees(pointing["angle_rad"]))
    grp.attrs["dither_amp_mpc"] = float(pointing["dither_amp_mpc"])
    grp.attrs["n_sources_drawn"] = int(sample["n_src_drawn"])
    (grp.attrs["keep_frac_occult"],
     grp.attrs["keep_frac_pointing"],
     grp.attrs["keep_frac_total"]) = (float(v) for v in sample["keep_fracs"])
    # Photo-z systematics, as scalars (z_true is deliberately not stored per
    # source). src_true_foreground_frac is the dilution: sources whose BPZ point
    # estimate cleared z_lens + 0.1 but whose true redshift is at or in front of
    # the lens, and which therefore carry no shear.
    grp.attrs["src_true_foreground_frac"] = float(sample["src_true_foreground_frac"])
    grp.attrs["src_z_true_mean"] = float(sample["src_z_true_mean"])

    # Intrinsic X-ray luminosity, for checking the sample against LoVoCCS's
    # selection. Written whether or not the mock observation was made.
    grp.attrs["lx500c_erg_s"] = float(sample["lx500c_erg_s"])
    grp.attrs["r500c_mpc_over_h"] = float(sample["r500c_mpc_over_h"])

    xray = sample.get("xray")
    if xray is not None:
        for name in ("xray_mosaic", "xray_counts", "xray_expmap", "xray_ideal",
                     "xray_mask"):
            grp.create_dataset(name, data=xray[name], compression="gzip")   # (G,G)
        grp.attrs["xray_valid_frac"] = float(np.mean(xray["xray_mask"]))
        # Fixed angular grid, so the physical scale is per sample.
        grp.attrs["xray_mpc_per_pix"] = float(xray["xray_mpc_per_pix"])
        grp.attrs["xray_arcsec_per_pix"] = float(xray["xray_arcsec_per_pix"])
        grp.attrs["xray_redshift"] = float(xray["xray_redshift"])
        grp.attrs["xray_fov_mpc"] = float(
            xray["xray_mpc_per_pix"] * CFG_XRAY.grid_size)
        # Archival depth varies per cluster, so it is a per-sample attribute.
        grp.attrs["xray_t_exp_ks"] = float(xray["xray_t_exp_ks"])


def write_sample_targets(f: h5py.File, sample_id: int, cubes, mass_log10, axes,
                         cube_stats, mass_mean, mass_std, axis_mean, axis_std):
    """
    Second pass: standardized cubes + global targets into the existing group.
    cubes and cube_stats are keyed on resolution: {N: raw (N,N,N)} and
    {N: (log10 mean, log10 std)}.
    """
    grp = f[f"{sample_id:06d}"]
    for N, cube in cubes.items():
        cube_mean, cube_std = cube_stats[N]
        cube_norm = log_standardize_with_floor(
            cube, cube_mean, cube_std, floor_value=CFG_GRID.floor_value)
        grp.create_dataset(f"density_cube_{N}", data=cube_norm,
                           compression="gzip")                                 # (Z,Y,X)

    std_mass = (float(mass_log10) - mass_mean) / mass_std
    std_axes = (axes.astype(np.float64) - axis_mean) / axis_std
    globals_target = np.concatenate(
        [np.array([std_mass], dtype=np.float32), std_axes.astype(np.float32)], axis=0)

    grp.create_dataset("cube_mass_log10_msun", data=np.float32(mass_log10))
    grp.create_dataset("axis_lengths_mpc", data=axes.astype(np.float32))
    grp.create_dataset("globals_target", data=globals_target)


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_FILENAME)
    if os.path.exists(out_path):
        os.remove(out_path)

    # Empirical n(z) template. The background selection is per sample (the
    # foreground cut moves with z_lens), so the template is kept whole and
    # sorted here; lovoccs_background_template slices it per sample.
    z_raw = np.load(CFG_LENS.empirical_redshift_npz_path)[CFG_LENS.empirical_redshift_key]
    z_template = np.asarray(z_raw, dtype=np.float64)
    z_template = np.sort(z_template[np.isfinite(z_template)])
    # The tightest cut any sample can apply, used only for the up-front check.
    z_lo_max = CFG_LENS.z_lens_max + CFG_LENS.src_z_foreground_pad
    n_worst = int(np.count_nonzero((z_template > z_lo_max)
                                   & (z_template < CFG_LENS.src_z_max)))
    if n_worst == 0:
        raise ValueError(
            f"empirical redshift template at {CFG_LENS.empirical_redshift_npz_path} "
            f"(key={CFG_LENS.empirical_redshift_key!r}) has no entries in "
            f"({z_lo_max}, {CFG_LENS.src_z_max}), so the highest-z lens would "
            f"have no background sources"
        )
    print(f"empirical n(z) [{CFG_LENS.empirical_template}]: {z_template.size} sources, "
          f"<z>={z_template.mean():.3f}, min={z_template.min():.3f}, "
          f"max={z_template.max():.3f}")
    print(f"  LoVoCCS background cuts: z_lens + {CFG_LENS.src_z_foreground_pad} "
          f"< z_s < {CFG_LENS.src_z_max}; {n_worst} sources "
          f"({100.0 * n_worst / z_template.size:.1f}%) survive at the worst-case "
          f"z_lens = {CFG_LENS.z_lens_max}")
    # TNG300 stellar mass / g-band pairs calibrating the BAHAMAS 'galaxy' cut.
    mag_calibration = load_tng_calibration()
    n_cal = int(np.sum(np.isfinite(mag_calibration[0]) & (mag_calibration[0] > 0)))
    print(f"galaxy cut: M_g <= {CFG_DATA.mag_cut_g} calibrated on "
          f"{'+'.join(CFG_DATA.tng_calibration_sims)} ({n_cal} subhalos with Mstar>0), "
          f"applied stochastically with seed={CFG_DATA.galaxy_cut_seed}")
    print(f"lens redshift: z_l ~ U[{CFG_LENS.z_lens_min}, {CFG_LENS.z_lens_max}] per sample, "
          f"keyed on (sim, cluster, projection) with z_lens_seed="
          f"{CFG_LENS.z_lens_seed}; n_source={CFG_LENS.n_source_per_arcmin2}/arcmin^2, "
          f"shape grid {CFG_LENS.shape_grid_size}x{CFG_LENS.shape_grid_size} "
          f"over +/-{CFG_GRID.fov_mpc} Mpc")
    if CFG_LENS.apply_photoz_error:
        print(f"  BPZ photo-z error: sigma_z/(1+z)={CFG_LENS.bpz_sigma_z}, "
              f"outlier rate {CFG_LENS.bpz_outlier_rate:.1%} redrawn from the "
              f"full n(z); lensing uses z_true, the catalog reports z_phot")
    else:
        print("  BPZ photo-z error: DISABLED (z_phot == z_true)")

    cluster_inds = np.array([f"{i:03d}" for i in range(*CFG_DATA.cluster_index_range)])
    sims = list(CFG_DATA.simulations)
    n_clusters_total = len(cluster_inds) * len(sims)
    n_proj = CFG_DATA.n_projections
    print(f"clusters={len(cluster_inds)}, sims={len(sims)}, proj/cluster={n_proj}, "
          f"total samples={n_clusters_total * n_proj}")

    resolutions = tuple(int(N) for N in CFG_GRID.resolutions)
    N_fine = max(resolutions)
    if any(N_fine % N for N in resolutions):
        raise ValueError(f"every resolution in {resolutions} must divide {N_fine}")
    print(f"resolutions {resolutions}: SPH (n_ngb={CFG_DENS.n_ngb}, h floor "
          f"{', '.join(f'{1e3 * sph_h_floor_mpc(s):.1f}' for s in sims)} kpc) "
          f"deposited at {N_fine}^3 and "
          f"block-averaged down; h_j cached under {CFG_DENS.cache_dir}")

    rng = np.random.default_rng(CFG_DATA.rng_seed)

    jobs = []
    for ci in cluster_inds:
        for sim in sims:
            npz_path = os.path.join(CFG_DATA.sim_data_root, sim, f"GrNm_{ci}.npz")
            for pi in range(n_proj):
                pv = rng.uniform(-1.0, 1.0, size=3)
                pv = pv / max(np.linalg.norm(pv), 1e-12)
                # proj_index identifies the sample to xray_photons.py: it keys
                # both the photon-list filename and z_lens. The viewing vector
                # and the per-sample seed stay on this rng stream -- the photon
                # list is a 3D object, so nothing about it depends on them.
                jobs.append((npz_path, sim, int(ci), pi, pv,
                             int(rng.integers(0, 2 ** 31 - 1))))

    check_photon_lists(jobs)

    # One task per (sim, cluster), so its projections share one h_j cache; see
    # build_cluster_samples. The jobs themselves are unchanged.
    tasks = {}
    for job in jobs:
        tasks.setdefault((job[1], job[2]), []).append(job)
    tasks = list(tasks.values())

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    n_workers = max(1, n_cpus // SPH_THREADS)
    print(f"using {n_workers} workers x {SPH_THREADS} SPH threads on {len(tasks)} "
          f"clusters; streaming samples to h5...")

    # Big ragged arrays are written the moment they arrive. The raw finest
    # cubes (8 MB each at 128^3, ~20 GB over the suite) go to a side file on
    # scratch rather than memory until the per-N stats exist; only the small
    # global targets are held.
    raw_path = os.path.join(CFG_DENS.cache_dir,
                            OUT_FILENAME.replace(".hdf5", "_raw_cubes.h5"))
    os.makedirs(CFG_DENS.cache_dir, exist_ok=True)
    held_targets = {}                 # sid -> (mass_log10, axes)
    cube_log_stats = {N: _LogStats() for N in resolutions}
    n_src_seen = []
    n_gal_seen = []

    with h5py.File(out_path, "w") as f_out, h5py.File(raw_path, "w") as f_raw:
        write_static_attrs(f_out)

        sid = 0
        with mp.get_context("spawn").Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(z_template, mag_calibration),
        ) as pool:
            for samples in pool.imap_unordered(build_cluster_samples, tasks, chunksize=1):
                for sample in samples:
                    write_sample_streaming(f_out, sid, sample)
                    rho = sample["raw_density_cube"]
                    f_raw.create_dataset(f"{sid:06d}", data=rho)
                    for N, cube in du.multires_cubes(rho, resolutions).items():
                        cube_log_stats[N].add(cube)
                    held_targets[sid] = (
                        float(sample["cube_mass_log10_msun"]),
                        np.asarray(sample["axis_lengths_mpc"]),
                    )
                    n_src_seen.append(int(sample["n_src"]))
                    n_gal_seen.append(int(sample["n_gal"]))
                    sid += 1

                    if sid % 250 == 0:
                        print(f"  built/wrote {sid}/{len(jobs)}", flush=True)

        n_src_seen = np.asarray(n_src_seen)
        print(f"source counts: min={n_src_seen.min()}, "
              f"median={int(np.median(n_src_seen))}, max={n_src_seen.max()}")
        n_gal_seen = np.asarray(n_gal_seen)
        print(f"member counts after the galaxy cut: min={n_gal_seen.min()}, "
              f"median={int(np.median(n_gal_seen))}, max={n_gal_seen.max()} "
              f"(max_nodes={CFG_DATA.max_nodes})")

        # -------------------------------
        # Normalization statistics over this dataset
        # -------------------------------
        # NOTE: these are BAHAMAS-only. When this file is combined with the
        # other simulations into a train/val/test split, whichever subset is
        # used for training should supply the stats, or they should be
        # recomputed across the combined training set.
        print("Computing normalization stats...")
        if not held_targets:
            raise RuntimeError("no samples were built")

        # Per resolution: a 128^3 cube resolves a denser core than a 16^3 one,
        # so the log10 distributions differ.
        cube_stats = {N: cube_log_stats[N].mean_std() for N in resolutions}

        mass_logs = np.array([m for m, _ in held_targets.values()], dtype=np.float64)
        axes_all = np.stack([a for _, a in held_targets.values()], axis=0).astype(np.float64)
        mass_mean = float(np.mean(mass_logs))
        mass_std = float(np.std(mass_logs) + 1e-6)
        axis_mean = np.mean(axes_all, axis=0)
        axis_std = np.std(axes_all, axis=0) + 1e-6

        for N in resolutions:
            print(f"3D density {N:3d}^3 log10 mean/std: "
                  f"{cube_stats[N][0]:.6f}, {cube_stats[N][1]:.6f}")
        print(f"Cube mass log10(Msun) mean/std: {mass_mean:.6f}, {mass_std:.6f}")
        print(f"Axis lengths Mpc mean (a,b,c) : "
              f"{axis_mean[0]:.6f}, {axis_mean[1]:.6f}, {axis_mean[2]:.6f}")
        print(f"Axis lengths Mpc std  (a,b,c) : "
              f"{axis_std[0]:.6f}, {axis_std[1]:.6f}, {axis_std[2]:.6f}")

        for N in resolutions:
            f_out.attrs[f"cube_log10_mean_{N}"] = cube_stats[N][0]
            f_out.attrs[f"cube_log10_std_{N}"] = cube_stats[N][1]
        f_out.attrs["cube_mass_log10_mean"] = mass_mean
        f_out.attrs["cube_mass_log10_std"] = mass_std
        f_out.attrs["axis_mean"] = axis_mean
        f_out.attrs["axis_std"] = axis_std
        f_out.attrs["n_samples"] = len(held_targets)

        # -------------------------------
        # Second pass: standardized cubes + global targets
        # -------------------------------
        print(f"Writing cubes/targets ({len(held_targets)} samples)...")
        for sid, (mass_log10, axes) in held_targets.items():
            cubes = du.multires_cubes(f_raw[f"{sid:06d}"][()], resolutions)
            write_sample_targets(f_out, sid, cubes, mass_log10, axes,
                                 cube_stats, mass_mean, mass_std,
                                 axis_mean, axis_std)

    os.remove(raw_path)
    print("Done.")
    print(f"{out_path} ({len(held_targets)} samples)")


if __name__ == "__main__":
    main()
