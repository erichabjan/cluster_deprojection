"""
Pure-numpy lensing utilities: cosmology, Smail n(z), KS93 forward/inverse, bilinear sampling,
2D binning. No jax dependency, so this module is importable from both `cl_dyn` and `jax_lense`.

Sign / Fourier conventions match jax_lensing.inversion (so KS-based truth from this module
round-trips exactly with the jax-lensing reconstruction in stage 2).
"""
import numpy as np
from typing import Tuple


C_LIGHT_KM_S = 2.99792458e5
# Newton's G in (km/s)^2 * Mpc / Msun
G_NEWTON_KM2_S2_MPC_MSUN = 4.3009125e-9


# ----- cosmology (flat LambdaCDM) -----

def comoving_distance_mpc(z, H0=70.0, Om0=0.3, n_steps=2048):
    """Comoving distance (Mpc) in flat LambdaCDM via trapezoidal quadrature."""
    z_arr = np.atleast_1d(z).astype(np.float64)
    out = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        if zi <= 0.0:
            out[i] = 0.0
            continue
        zg = np.linspace(0.0, zi, n_steps)
        Ez = np.sqrt(Om0 * (1.0 + zg) ** 3 + (1.0 - Om0))
        out[i] = (C_LIGHT_KM_S / H0) * np.trapz(1.0 / Ez, zg)
    return out if np.ndim(z) > 0 else float(out[0])


def angular_diameter_distance_mpc(z, H0=70.0, Om0=0.3):
    return comoving_distance_mpc(z, H0=H0, Om0=Om0) / (1.0 + np.asarray(z))


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


def beta_factor(z_l, z_s, H0=70.0, Om0=0.3):
    """beta(z_l, z_s) = D_ls / D_s * Heaviside(z_s - z_l). Vectorized over z_s."""
    z_s = np.atleast_1d(z_s).astype(np.float64)
    Ds = angular_diameter_distance_mpc(z_s, H0=H0, Om0=Om0)
    Dls = angular_diameter_distance_z1z2_mpc(z_l, z_s, H0=H0, Om0=Om0)
    out = np.where(z_s > z_l, Dls / np.clip(Ds, 1e-6, None), 0.0)
    return out


# ----- Smail n(z) -----

def smail_pdf(z, alpha=2.0, beta=1.5, z0=0.7):
    z = np.asarray(z)
    return np.where(z > 0, (z ** alpha) * np.exp(-((z / z0) ** beta)), 0.0)


def sample_smail_redshifts(n_samples, alpha=2.0, beta=1.5, z0=0.7, z_max=5.0, rng=None):
    """Inverse-CDF sampling on a fine grid."""
    if rng is None:
        rng = np.random.default_rng()
    zg = np.linspace(0.0, z_max, 4096)
    pdf = smail_pdf(zg, alpha, beta, z0)
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    u = rng.uniform(size=n_samples)
    return np.interp(u, cdf, zg)


def mean_beta_over_smail(z_l, alpha=2.0, beta=1.5, z0=0.7, z_max=5.0,
                         H0=70.0, Om0=0.3, n_grid=4096):
    """<beta> over n(z); used to convert kappa_inf <-> recovered kappa_eff."""
    zg = np.linspace(z_l + 1e-3, z_max, n_grid)
    nz = smail_pdf(zg, alpha, beta, z0)
    bz = beta_factor(z_l, zg, H0=H0, Om0=Om0)
    return float(np.trapz(bz * nz, zg) / np.trapz(nz, zg))


# ----- KS93 forward/inverse, numpy version (matches jax_lensing.inversion) -----

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


def ks93_numpy(g1: np.ndarray, g2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """shear (g1, g2) -> convergence (kE, kB). Mirrors jax_lensing.inversion.ks93."""
    assert g1.shape == g2.shape and g1.shape[0] == g1.shape[1]
    N = g1.shape[0]
    p1, p2, k2sq = _ks_kernel(N)
    g1hat = np.fft.fft2(g1)
    g2hat = np.fft.fft2(g2)
    kEhat = (p1 * g1hat + p2 * g2hat) / k2sq
    kBhat = -(p2 * g1hat - p1 * g2hat) / k2sq
    return np.fft.ifft2(kEhat).real, np.fft.ifft2(kBhat).real


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


def bin2d_mean(x, y, w, fov, N):
    s = bin2d_sum(x, y, w, fov, N)
    c = bin2d_count(x, y, fov, N)
    out = np.zeros_like(s, dtype=np.float32)
    valid = c > 0
    out[valid] = s[valid] / c[valid]
    return out, c


# ----- pooling -----

def avg_pool_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr
    H, W = arr.shape
    assert H % factor == 0 and W % factor == 0, f"avg_pool_2d: ({H},{W}) not divisible by {factor}"
    return arr.reshape(H // factor, factor, W // factor, factor).mean(axis=(1, 3))


def avg_pool_3d(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr
    Z, Y, X = arr.shape
    assert Z % factor == 0 and Y % factor == 0 and X % factor == 0, \
        f"avg_pool_3d: ({Z},{Y},{X}) not divisible by {factor}"
    return arr.reshape(Z // factor, factor, Y // factor, factor, X // factor, factor).mean(axis=(1, 3, 5))
