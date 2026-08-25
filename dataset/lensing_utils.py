"""
Pure-numpy lensing utilities for the shape_dynamics dataset: flat-LambdaCDM
distances, critical surface densities, empirical source-redshift sampling,
KS93 inverse (kappa -> gamma), bilinear sampling, 2D binning, intrinsic
ellipticities, and the two survey masks (member occultation and the dithered
pointing footprint).

Trimmed from jaxlense_dataset/lensing_utils.py to only what
draw_shape_catalog and the Sigma -> kappa -> gamma construction need.
Sign / Fourier conventions match jax_lensing.inversion.
"""
import numpy as np
from typing import Tuple
from scipy.spatial import cKDTree


C_LIGHT_KM_S = 2.99792458e5
# Newton's G in (km/s)^2 * Mpc / Msun
G_NEWTON_KM2_S2_MPC_MSUN = 4.3009125e-9


# ----- cosmology (flat LambdaCDM) -----

def comoving_distance_mpc(z, H0=70.0, Om0=0.3, n_steps=2048):
    """
    Comoving distance (Mpc) in flat LambdaCDM via trapezoidal quadrature.
    Vectorized: one cumulative integral on a shared grid up to max(z), then
    interpolated to each element (avoids a per-element Python loop, which is
    prohibitive for the ~10^5-10^6 source catalogs drawn per sample).
    """
    z_arr = np.atleast_1d(z).astype(np.float64)
    z_max = float(z_arr.max(initial=0.0))
    if z_max <= 0.0:
        out = np.zeros_like(z_arr)
    else:
        zg = np.linspace(0.0, z_max, n_steps)
        inv_Ez = 1.0 / np.sqrt(Om0 * (1.0 + zg) ** 3 + (1.0 - Om0))
        Dc_grid = np.concatenate([
            [0.0],
            np.cumsum(0.5 * (inv_Ez[1:] + inv_Ez[:-1]) * np.diff(zg)),
        ]) * (C_LIGHT_KM_S / H0)
        out = np.where(z_arr > 0.0, np.interp(z_arr, zg, Dc_grid), 0.0)
    return out if np.ndim(z) > 0 else float(out[0])


def angular_diameter_distance_mpc(z, H0=70.0, Om0=0.3):
    return comoving_distance_mpc(z, H0=H0, Om0=Om0) / (1.0 + np.asarray(z))


def mpc_per_arcmin(z, H0=70.0, Om0=0.3):
    """Physical Mpc subtended by one arcmin at redshift z."""
    return angular_diameter_distance_mpc(z, H0=H0, Om0=Om0) * (np.pi / 180.0 / 60.0)


def angular_diameter_distance_z1z2_mpc(z1, z2, H0=70.0, Om0=0.3):
    """Flat LambdaCDM A.D. distance from z1 (scalar) to z2 (vector)."""
    z2 = np.atleast_1d(z2).astype(np.float64)
    Dc1 = comoving_distance_mpc(z1, H0=H0, Om0=Om0)
    Dc2 = comoving_distance_mpc(z2, H0=H0, Om0=Om0)
    return (Dc2 - Dc1) / (1.0 + z2)


def sigma_crit_inf_msun_per_mpc2(z_l, H0=70.0, Om0=0.3):
    """
    Sigma_crit at source-plane infinity = limit of c^2/(4 pi G) * D_s / (D_l * D_ls) as z_s -> inf.
    Reduces to c^2/(4 pi G) / D_l. Returns Msun / Mpc^2.
    """
    D_l = angular_diameter_distance_mpc(z_l, H0=H0, Om0=Om0)
    return (C_LIGHT_KM_S ** 2) / (4.0 * np.pi * G_NEWTON_KM2_S2_MPC_MSUN * D_l)


def sigma_crit_msun_per_mpc2(z_l, z_s, H0=70.0, Om0=0.3):
    """
    Per-source critical surface density:
        Sigma_crit(z_l, z_s) = c^2 / (4 pi G) * D_s / (D_l * D_ls).
    Returns Msun/Mpc^2. Foreground sources (z_s <= z_l) get +inf so that
    1/Sigma_crit -> 0, i.e., they feel no lensing.
    """
    z_s = np.atleast_1d(z_s).astype(np.float64)
    D_l = angular_diameter_distance_mpc(z_l, H0=H0, Om0=Om0)
    D_s = angular_diameter_distance_mpc(z_s, H0=H0, Om0=Om0)
    D_ls = angular_diameter_distance_z1z2_mpc(z_l, z_s, H0=H0, Om0=Om0)
    pref = (C_LIGHT_KM_S ** 2) / (4.0 * np.pi * G_NEWTON_KM2_S2_MPC_MSUN)
    denom = D_l * D_ls
    return np.where(z_s > z_l,
                    pref * D_s / np.clip(denom, 1e-30, None),
                    np.inf)


# ----- empirical n(z) sampling -----

def sample_empirical_redshifts(n_samples, z_template, rng=None):
    """Bootstrap n_samples redshifts (with replacement) from z_template."""
    if rng is None:
        rng = np.random.default_rng()
    if z_template.size == 0:
        raise ValueError("sample_empirical_redshifts: z_template is empty")
    return rng.choice(z_template, size=n_samples, replace=True)


# ----- intrinsic ellipticities -----

def truncated_sigma(sigma_in):
    """
    Per-component dispersion actually realized when a 2D Gaussian of width
    sigma_in is truncated to the physical range |e| < 1:

        sigma_out^2 = sigma_in^2 * [1 - (1+a) e^-a] / [1 - e^-a],  a = 1/(2 sigma_in^2)

    (integrate the Rayleigh |e| distribution over [0, 1)).
    """
    a = 1.0 / (2.0 * sigma_in ** 2)
    ratio = (1.0 - (1.0 + a) * np.exp(-a)) / (1.0 - np.exp(-a))
    return sigma_in * np.sqrt(ratio)


def gaussian_sigma_for_truncated(sigma_target, tol=1e-10, max_iter=200):
    """
    Invert truncated_sigma: the Gaussian width to draw from so that, after
    rejecting |e| >= 1, the catalog realizes sigma_target per component.

    sigma_target must be below 1/sqrt(2), the largest dispersion any
    distribution on the unit disk can have.
    """
    if not 0.0 < sigma_target < 1.0 / np.sqrt(2.0):
        raise ValueError(
            f"sigma_target={sigma_target} is outside (0, 1/sqrt(2)); no "
            f"distribution on the unit disk has that per-component dispersion."
        )
    lo, hi = sigma_target, 10.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if truncated_sigma(mid) < sigma_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def draw_intrinsic_ellipticity(n, sigma_in, rng=None):
    """
    Complex intrinsic ellipticities, Gaussian per component, rejecting draws
    outside the unit disk.

    The truncation is not cosmetic. e = (a-b)/(a+b) has no meaning beyond
    |e| = 1, and the reduced-shear transformation
    (e_int + g)/(1 + conj(g) e_int) has a pole at |e_int| = 1/|g|, which sits
    just a few sigma away once sigma approaches the boundary. At sigma = 0.4,
    exp(-1/2 sigma^2) = 4.4% of an untruncated Gaussian would land outside.
    """
    if rng is None:
        rng = np.random.default_rng()
    e = (rng.normal(0.0, sigma_in, size=n)
         + 1j * rng.normal(0.0, sigma_in, size=n))
    bad = np.abs(e) >= 1.0
    while np.any(bad):
        n_bad = int(bad.sum())
        e[bad] = (rng.normal(0.0, sigma_in, size=n_bad)
                  + 1j * rng.normal(0.0, sigma_in, size=n_bad))
        bad = np.abs(e) >= 1.0
    return e


# ----- survey masks -----

def occultation_keep_mask(src_x, src_y, mem_x, mem_y, radii_mpc):
    """
    Drop sources landing behind a cluster member, where the blend makes a shape
    unmeasurable. Members are discs of radii_mpc in the projected plane; every
    retained source is behind the lens, so this is purely geometric.

    radii_mpc is per member (the BCG is larger than the rest), and the queries
    are grouped by distinct radius, of which there are only two.
    """
    keep = np.ones(src_x.shape, dtype=bool)
    if src_x.size == 0 or mem_x.size == 0:
        return keep

    tree = cKDTree(np.column_stack([src_x, src_y]))
    radii_mpc = np.broadcast_to(np.asarray(radii_mpc, dtype=float), mem_x.shape)

    for r in np.unique(radii_mpc):
        sel = radii_mpc == r
        hit = tree.query_ball_point(np.column_stack([mem_x[sel], mem_y[sel]]), r)
        for idx in hit:
            if idx:
                keep[idx] = False
    return keep


def draw_pointing(rng, z_lens, side_arcmin_range, dither_amp_arcmin_range,
                  n_exposures, scale_scatter=0.0, center_offset_frac=0.0,
                  H0=70.0, Om0=0.3):
    """
    A rectangular survey pointing, dithered n_exposures times.

    The sides are drawn in arcmin, because a field of view is an angular
    quantity, and converted to Mpc at the lens redshift; the conversion carries
    a lognormal scatter, so the redshift dependence is noisy rather than exact
    (surveys differ in depth, tiling and usable area). Returns everything needed
    to reconstruct the footprint later.
    """
    scale = mpc_per_arcmin(z_lens, H0=H0, Om0=Om0) * np.exp(
        rng.normal(0.0, scale_scatter) if scale_scatter > 0 else 0.0)

    sides = rng.uniform(*side_arcmin_range, size=2) * scale
    angle = float(rng.uniform(0.0, np.pi))
    center = rng.uniform(-1.0, 1.0, size=2) * center_offset_frac * sides
    amp = float(rng.uniform(*dither_amp_arcmin_range)) * scale
    offsets = rng.uniform(-amp, amp, size=(int(n_exposures), 2))

    return {
        "center_mpc": center.astype(np.float64),
        "sides_mpc": sides.astype(np.float64),
        "angle_rad": angle,
        "offsets_mpc": offsets.astype(np.float64),
        "dither_amp_mpc": amp,
        "mpc_per_arcmin": scale,
    }


def pointing_coverage_fraction(x, y, pointing):
    """
    Fraction of the exposures covering each (x, y): 0 outside the footprint, 1
    where every exposure overlaps, and a ramp of width ~2 * dither amplitude
    around the edges, where only some exposures reach.
    """
    x = np.atleast_1d(x).astype(np.float64)
    y = np.atleast_1d(y).astype(np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)

    cx, cy = pointing["center_mpc"]
    ca, sa = np.cos(pointing["angle_rad"]), np.sin(pointing["angle_rad"])
    dx, dy = x - cx, y - cy
    xr = ca * dx + sa * dy          # into the pointing's own frame
    yr = -sa * dx + ca * dy

    hx, hy = 0.5 * pointing["sides_mpc"]
    ox, oy = pointing["offsets_mpc"][:, 0], pointing["offsets_mpc"][:, 1]

    inside = ((np.abs(xr[:, None] - ox[None, :]) <= hx)
              & (np.abs(yr[:, None] - oy[None, :]) <= hy))
    return inside.mean(axis=1)


def coverage_map(pointing, fov, N):
    """pointing_coverage_fraction on the (N, N) grid over [-fov, fov], as [iy, ix]."""
    centers = (np.arange(N) + 0.5) / N * (2.0 * fov) - fov
    gx, gy = np.meshgrid(centers, centers)          # gy varies along axis 0
    cov = pointing_coverage_fraction(gx.ravel(), gy.ravel(), pointing)
    return cov.reshape(N, N).astype(np.float32)


# ----- KS93 inverse, numpy version (matches jax_lensing.inversion) -----

def _ks_kernel(N):
    k1 = np.fft.fftfreq(N)
    k2 = np.fft.fftfreq(N)
    K1, K2 = np.meshgrid(k1, k2)
    p1 = K1 * K1 - K2 * K2
    p2 = 2.0 * K1 * K2
    k2sq = K1 * K1 + K2 * K2
    k2sq[0, 0] = 1.0
    return p1, p2, k2sq


def ks93_inv_numpy(kE: np.ndarray, kB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """convergence (E, B) -> shear (g1, g2). Mirrors jax_lensing.inversion.ks93inv."""
    assert kE.shape == kB.shape and kE.shape[0] == kE.shape[1]
    N = kE.shape[0]
    p1, p2, k2sq = _ks_kernel(N)
    kEhat = np.fft.fft2(kE)
    kBhat = np.fft.fft2(kB)
    g1hat = (p1 * kEhat - p2 * kBhat) / k2sq
    g2hat = (p2 * kEhat + p1 * kBhat) / k2sq
    return np.fft.ifft2(g1hat).real, np.fft.ifft2(g2hat).real


# ----- bilinear sampling on a regular grid -----

def bilinear_sample(grid: np.ndarray, x_world: np.ndarray, y_world: np.ndarray, fov: float):
    """Bilinearly sample `grid` (shape (N, N), indexed [iy, ix]) at world coords (x, y) in [-fov, fov]."""
    N = grid.shape[0]
    pix_x = (x_world + fov) / (2.0 * fov) * (N - 1)
    pix_y = (y_world + fov) / (2.0 * fov) * (N - 1)
    pix_x = np.clip(pix_x, 0.0, N - 1)
    pix_y = np.clip(pix_y, 0.0, N - 1)
    x0 = np.floor(pix_x).astype(np.int64)
    y0 = np.floor(pix_y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, N - 1)
    y1 = np.clip(y0 + 1, 0, N - 1)
    fx = pix_x - x0
    fy = pix_y - y0
    g00 = grid[y0, x0]; g10 = grid[y0, x1]
    g01 = grid[y1, x0]; g11 = grid[y1, x1]
    return ((1.0 - fy) * ((1.0 - fx) * g00 + fx * g10)
            + fy * ((1.0 - fx) * g01 + fx * g11))


# ----- 2D binning helpers -----

def bin2d_count(x, y, fov, N):
    H, _, _ = np.histogram2d(y.astype(np.float64), x.astype(np.float64),
                             bins=(N, N), range=((-fov, fov), (-fov, fov)))
    return H.astype(np.float32)


def bin2d_sum(x, y, w, fov, N):
    H, _, _ = np.histogram2d(y.astype(np.float64), x.astype(np.float64),
                             bins=(N, N), range=((-fov, fov), (-fov, fov)),
                             weights=w.astype(np.float64))
    return H.astype(np.float32)
