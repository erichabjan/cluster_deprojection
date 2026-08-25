#!/usr/bin/env python3
"""
Build the BAHAMAS dataset for the conditional diffusion model.

Writes a single bahamas_dataset.hdf5 covering every (simulation, cluster,
projection). It holds no train/val split: the split is made later, across the
per-simulation-suite files, once the other suites have been run through the same
analysis.

Per (cluster, sim, projection) sample, all in ONE viewing frame (a single
projection vector rotates both the cluster members and the matter particles):
  - dynamics conditioning images (3, H, W): gal_xy, gal_z_xy, gal_z_disp_xy
  - cluster-member point cloud, padded to max_nodes (features/targets/pixel coords/mask)
  - 3D density target cube (log-standardized with this file's stats) + global targets
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

    Each sample stores, ragged (no padding, variable N):
      src_features (N, 8) float32: e1, e2, ix, iy, z_s, z_lens, cell_frac, coverage
        - ix, iy: integer cell coordinates on the shape grid (stored as float32)
        - cell_frac: (# sources in that cell) / N
        - coverage: fraction of exposures covering that source
      src_cell_id  (N,)   int32: iy * shape_grid_size + ix; rows are PRE-SORTED by this
      src_cell_ptr (S*S+1,) int64: CSR pointers, cell c's sources are rows ptr[c]:ptr[c+1]
      src_coverage (S, S)  float32: the same coverage fraction on the shape grid,
        so an empty cell is distinguishable from an unobserved one

  - NEW: a mock Chandra ACIS-I X-ray mosaic of the same cluster, in the same
    viewing frame. xray_photons.py has already made the pyXSIM photon list for
    each (sim, cluster, redshift bin); each sample projects the list nearest its
    z_lens along its own viewing direction and observes it with a 3x3 ACIS-I
    mosaic. Stored on a fixed 52' *angular* grid, so the physical scale is a
    per-sample attribute (xray_mpc_per_pix):
      xray_mosaic (G,G): exposure-corrected 0.5-2 keV surface brightness
      xray_counts (G,G): mosaic counts
      xray_expmap (G,G): mean exposure map, cm^2 s
      xray_ideal  (G,G): the same band with no instrument, PSF or background

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
from config import CFG_GRID, CFG_SCALE, CFG_LENS, CFG_COSMO, CFG_DATA, CFG_XRAY
import lensing_utils as lu

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

    # Intrinsic ellipticities, drawn wide and truncated to the physical |e| < 1
    # so the catalog realizes CFG_LENS.sigma_e_realized per component.
    e_int = lu.draw_intrinsic_ellipticity(g_complex.shape[0], SIGMA_E_INPUT, rng=rng)
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


def z_lens_for_seed(seed) -> float:
    """
    The lens redshift build_one_sample will draw for this job. It is that rng's
    first call, so a fresh generator on the same seed reproduces it; used to
    check the photon lists exist before a long run starts.
    """
    return float(np.random.default_rng(seed).uniform(
        CFG_LENS.z_lens_min, CFG_LENS.z_lens_max))


def check_photon_lists(jobs):
    """
    Fail before the build starts, not two hours in, if xray_photons.slurm has
    not covered every (sim, cluster, redshift bin) the sample list needs.
    """
    if not CFG_XRAY.enabled:
        print("X-ray maps disabled (CFG_XRAY.enabled = False)")
        return

    xu = _xray_module()
    needed, missing = set(), set()
    for _, sim, cluster_idx, _, seed in jobs:
        z_bin = xu.nearest_z_bin(z_lens_for_seed(seed))
        key = (sim, cluster_idx, z_bin)
        if key in needed:
            continue
        needed.add(key)
        if not os.path.exists(xu.photon_prefix(sim, cluster_idx, z_bin) + ".h5"):
            missing.add(key)

    if missing:
        examples = "\n  ".join(
            f"{s} GrNm_{c:03d} z={z:.4f}" for s, c, z in sorted(missing)[:5])
        raise FileNotFoundError(
            f"{len(missing)} of {len(needed)} photon lists are missing from "
            f"{CFG_XRAY.photon_dir}, e.g.:\n  {examples}\n"
            f"Run xray_photons.slurm first, or set CFG_XRAY.enabled = False."
        )
    print(f"X-ray: {len(needed)} photon lists present in {CFG_XRAY.photon_dir}; "
          f"{CFG_XRAY.n_side}x{CFG_XRAY.n_side} {CFG_XRAY.instrument} mosaic, "
          f"{CFG_XRAY.t_exp_ks} ks per pointing, "
          f"{CFG_XRAY.band_emin_kev}-{CFG_XRAY.band_emax_kev} keV, stored "
          f"{CFG_XRAY.grid_size}^2 over {CFG_XRAY.fov_arcmin}'")


def build_xray_maps(sim, cluster_idx, proj_vec, z_lens, h_sim, seed):
    """
    Project this sample's photon list along its viewing direction and observe it
    with a 3x3 ACIS-I mosaic. ~2 min per sample, and by far the most expensive
    thing in build_one_sample.

    make_photons already ran in xray_photons.py, once per (sim, cluster,
    redshift bin); here we pick the bin nearest z_lens and do the parts that
    depend on the projection.
    """
    if not CFG_XRAY.enabled:
        return None

    xu = _xray_module()
    z_bin = xu.nearest_z_bin(z_lens)
    photon_prefix = xu.photon_prefix(sim, cluster_idx, z_bin)
    if not os.path.exists(photon_prefix + ".h5"):
        raise FileNotFoundError(
            f"no photon list for {sim} GrNm_{cluster_idx:03d} at z={z_bin:.4f} "
            f"({photon_prefix}.h5). Run xray_photons.slurm first, or set "
            f"CFG_XRAY.enabled = False."
        )

    # Relative to the worker's CWD (CFG_XRAY.work_dir, set in _init_worker), so
    # SOXS's SIMPUT/photon-list cross-references resolve the way they do in the
    # notebook. The seed makes it unique across samples.
    work_prefix = f"s_{sim}_{cluster_idx:03d}_{seed}"
    try:
        products = xu.project_and_observe(photon_prefix, work_prefix, proj_vec)
        maps = xu.build_sample_maps(products, z_bin, h_sim)
    finally:
        xu.clean_intermediates(work_prefix)     # never touches the photon list
    return maps


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
    # Mean channel: per-pixel mean of the standardized member redshift (empty
    # pixels stay at 0). Dispersion channel: per-pixel redshift dispersion on
    # its own scale, which is ~z_std/8 and would otherwise be a flat image.
    gal_z_xy = make_galaxy_los_mean_xy(x, y, z_gal_s, fov, N_img)
    gal_z_disp_xy = (make_galaxy_los_disp_xy(x, y, z_gal.astype(np.float32),
                                            fov, N_img) / CFG_SCALE.z_disp_std)

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

    N_gal = x.shape[0]
    n_feat = np.full((N_gal,), float(N_gal), dtype=np.float32)
    gal_features = np.stack([x_s, y_s, z_gal_s, n_feat], axis=-1).astype(np.float32)
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

    src_features, src_cell_id, src_cell_ptr = process_source_catalog(
        cat, z_lens, fov, S)

    # -------------------------------
    # Mock Chandra ACIS-I mosaic, in this same viewing frame
    # -------------------------------
    xray = build_xray_maps(sim, cluster_idx, proj_vec, z_lens, h_sim, seed)

    return dict(
        sim=str(sim),
        cluster_idx=np.int32(cluster_idx),
        halo_mass=np.float32(data["Mfof"]),
        proj_vec=np.asarray(proj_vec, dtype=np.float32),
        z_lens=np.float32(z_lens),

        gal_xy=gal_xy.astype(np.float32),
        gal_z_xy=gal_z_xy.astype(np.float32),
        gal_z_disp_xy=gal_z_disp_xy.astype(np.float32),

        gal_features=feat_pad,
        gal_targets=targ_pad,
        gal_pixel_coords=pix_pad,
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

        raw_density_cube=density_cube,
        cube_mass_log10_msun=np.float32(cube_mass_log10_msun),
        axis_lengths_mpc=axis_lengths_mpc,

        xray=xray,
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
    f.attrs["n_source_per_arcmin2"] = CFG_LENS.n_source_per_arcmin2
    f.attrs["sigma_e_realized"] = CFG_LENS.sigma_e_realized
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
    f.attrs["gal_feature_columns"] = np.array([b"x", b"y", b"z_gal", b"Ngal"], dtype="S8")
    f.attrs["gal_target_columns"] = np.array([b"z", b"vx", b"vy"], dtype="S8")
    f.attrs["gal_pixel_coords_columns"] = np.array([b"x_pix", b"y_pix"], dtype="S8")
    f.attrs["globals_target_columns"] = np.array(
        [b"mass_log10_msun", b"axis_a_mpc", b"axis_b_mpc", b"axis_c_mpc"], dtype="S20")

    # Mock X-ray mosaics: same viewing frame as the images above, but on their
    # own fixed *angular* grid, so xray_mpc_per_pix is a per-sample attribute.
    if CFG_XRAY.enabled:
        f.attrs["xray_grid_size"] = CFG_XRAY.grid_size
        f.attrs["xray_fov_arcmin"] = CFG_XRAY.fov_arcmin
        f.attrs["xray_instrument"] = CFG_XRAY.instrument
        f.attrs["xray_n_side"] = CFG_XRAY.n_side
        f.attrs["xray_step_arcmin"] = CFG_XRAY.step_arcmin
        f.attrs["xray_t_exp_ks"] = CFG_XRAY.t_exp_ks
        f.attrs["xray_band_kev"] = np.array(
            [CFG_XRAY.band_emin_kev, CFG_XRAY.band_emax_kev], dtype=np.float64)
        f.attrs["xray_z_bin_width"] = CFG_XRAY.z_bin_width
        f.attrs["xray_nH_gal"] = CFG_XRAY.nH_gal
        f.attrs["xray_Z_met"] = CFG_XRAY.Z_met
        f.attrs["xray_maps"] = np.array(
            [b"xray_mosaic", b"xray_counts", b"xray_expmap", b"xray_ideal"], dtype="S16")
        f.attrs["xray_mosaic_units"] = "counts s^-1 cm^-2 arcsec^-2 (exposure corrected)"
        f.attrs["xray_ideal_units"] = (
            "counts s^-1 cm^-2 arcsec^-2, no instrument, PSF or background")
        f.attrs["xray_expmap_units"] = (
            "cm^2 (SOXS-normalised, i.e. exposure map / exposure time), "
            "mean over the stored pixel")


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
        [sample["gal_xy"], sample["gal_z_xy"], sample["gal_z_disp_xy"]],
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

    xray = sample.get("xray")
    if xray is not None:
        for name in ("xray_mosaic", "xray_counts", "xray_expmap", "xray_ideal"):
            grp.create_dataset(name, data=xray[name], compression="gzip")   # (G,G)
        # Fixed angular grid, so the physical scale is per sample.
        grp.attrs["xray_mpc_per_pix"] = float(xray["xray_mpc_per_pix"])
        grp.attrs["xray_arcsec_per_pix"] = float(xray["xray_arcsec_per_pix"])
        grp.attrs["xray_redshift"] = float(xray["xray_redshift"])
        grp.attrs["xray_fov_mpc"] = float(
            xray["xray_mpc_per_pix"] * CFG_XRAY.grid_size)


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
    out_path = os.path.join(OUT_DIR, OUT_FILENAME)
    if os.path.exists(out_path):
        os.remove(out_path)

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
    # TNG300 stellar mass / g-band pairs calibrating the BAHAMAS 'galaxy' cut.
    mag_calibration = load_tng_calibration()
    n_cal = int(np.sum(np.isfinite(mag_calibration[0]) & (mag_calibration[0] > 0)))
    print(f"galaxy cut: M_g <= {CFG_DATA.mag_cut_g} calibrated on "
          f"{'+'.join(CFG_DATA.tng_calibration_sims)} ({n_cal} subhalos with Mstar>0), "
          f"applied stochastically with seed={CFG_DATA.galaxy_cut_seed}")
    print(f"lens redshift: z_l ~ U[{CFG_LENS.z_lens_min}, {CFG_LENS.z_lens_max}] per sample; "
          f"n_source={CFG_LENS.n_source_per_arcmin2}/arcmin^2, "
          f"shape grid {CFG_LENS.shape_grid_size}x{CFG_LENS.shape_grid_size} "
          f"over +/-{CFG_GRID.fov_mpc} Mpc")

    cluster_inds = np.array([f"{i:03d}" for i in range(*CFG_DATA.cluster_index_range)])
    sims = list(CFG_DATA.simulations)
    n_clusters_total = len(cluster_inds) * len(sims)
    n_proj = CFG_DATA.n_projections
    print(f"clusters={len(cluster_inds)}, sims={len(sims)}, proj/cluster={n_proj}, "
          f"total samples={n_clusters_total * n_proj}")

    rng = np.random.default_rng(CFG_DATA.rng_seed)

    jobs = []
    for ci in cluster_inds:
        for sim in sims:
            npz_path = os.path.join(CFG_DATA.sim_data_root, sim, f"GrNm_{ci}.npz")
            for _ in range(n_proj):
                pv = rng.uniform(-1.0, 1.0, size=3)
                pv = pv / max(np.linalg.norm(pv), 1e-12)
                jobs.append((npz_path, sim, int(ci), pv,
                             int(rng.integers(0, 2 ** 31 - 1))))

    check_photon_lists(jobs)

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"using {n_workers} workers; streaming samples to h5...")

    # Big ragged arrays are written the moment they arrive; only the small
    # cubes + global targets are held back until the dataset stats exist.
    held_targets = {}                 # sid -> (cube, mass_log10, axes)
    n_src_seen = []
    n_gal_seen = []

    with h5py.File(out_path, "w") as f_out:
        write_static_attrs(f_out)

        with mp.get_context("spawn").Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(z_template, mag_calibration),
        ) as pool:
            for sid, sample in enumerate(
                    pool.imap_unordered(build_one_sample, jobs, chunksize=1)):
                write_sample_streaming(f_out, sid, sample)
                held_targets[sid] = (
                    sample["raw_density_cube"],
                    float(sample["cube_mass_log10_msun"]),
                    np.asarray(sample["axis_lengths_mpc"]),
                )
                n_src_seen.append(int(sample["n_src"]))
                n_gal_seen.append(int(sample["n_gal"]))

                if (sid + 1) % 250 == 0:
                    print(f"  built/wrote {sid + 1}/{len(jobs)}", flush=True)

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

        cube_logs = []
        for cube, _, _ in held_targets.values():
            cmask = cube > 0.0
            if np.any(cmask):
                cube_logs.append(np.log10(cube[cmask].astype(np.float64)))
        if not cube_logs:
            raise RuntimeError("no positive-valued cube voxels found")
        cube_logs = np.concatenate(cube_logs)
        cube_mean = float(np.mean(cube_logs))
        cube_std = float(np.std(cube_logs) + 1e-6)

        mass_logs = np.array([m for _, m, _ in held_targets.values()], dtype=np.float64)
        axes_all = np.stack([a for _, _, a in held_targets.values()], axis=0).astype(np.float64)
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

        f_out.attrs["cube_log10_mean"] = cube_mean
        f_out.attrs["cube_log10_std"] = cube_std
        f_out.attrs["cube_mass_log10_mean"] = mass_mean
        f_out.attrs["cube_mass_log10_std"] = mass_std
        f_out.attrs["axis_mean"] = axis_mean
        f_out.attrs["axis_std"] = axis_std
        f_out.attrs["n_samples"] = len(held_targets)

        # -------------------------------
        # Second pass: standardized cubes + global targets
        # -------------------------------
        print(f"Writing cubes/targets ({len(held_targets)} samples)...")
        for sid, (cube, mass_log10, axes) in held_targets.items():
            write_sample_targets(f_out, sid, cube, mass_log10, axes,
                                 cube_mean, cube_std, mass_mean, mass_std,
                                 axis_mean, axis_std)

    print("Done.")
    print(f"{out_path} ({len(held_targets)} samples)")


if __name__ == "__main__":
    main()
