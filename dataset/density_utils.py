"""
Adaptive-SPH density cubes for the dataset builds, shared by every suite.

Ported from /scratch/habjan.e/density_smoothing_tests/estimators.py (the sph64
estimator there). The field is

    rho(x) = sum_j m_j W(|x - x_j|, h_j),

with W the Gadget cubic spline of compact support h_j, and h_j the distance
from particle j to its n_ngb-th nearest neighbour over all species together,
floored at the spline-equivalent of the gravitational softening. Every
particle carries its own h_j, so the kernel is adaptive: a few tens of kpc in
the core, a few hundred at the cube's edge.

A voxel stores (1 / V_vox) * the integral of rho over the voxel. Each kernel is
sampled on a sub-voxel grid aligned to the voxel edges, fine enough to resolve
it (>= sub samples across its support), and the samples are renormalised to
the particle's exact mass, so mass is conserved particle by particle; mass that
falls outside the cube is tallied rather than dropped silently. A kernel
smaller than one sub-cell goes whole into the voxel holding its centre.

The field is deposited once, at the finest resolution, and the coarser cubes
are block averages of it (block_average). Voxel integrals add exactly and the
grids nest, so a 16^3 cube made this way is the same field integrated over
larger voxels -- and more accurate than a direct 16^3 deposit, whose sub-voxel
sampling is coarser.

h_j depends only on the distances between particles, which a rotation leaves
unchanged, so it is the same for every projection of a cluster. It is computed
once per (simulation, cluster) and cached on scratch (cached_smoothing_lengths).
"""
import math
import os

import numba as nb
import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# Smoothing lengths
# ============================================================

def smoothing_lengths(pos, n_ngb, h_floor_mpc, workers=1):
    """
    Per-particle SPH support h_j (Mpc): distance to the n_ngb-th nearest
    neighbour among all particles in `pos`, floored at h_floor_mpc.

    pos must hold every particle of the cutout, not just those inside the cube,
    or the particles near the cube's faces would see only half their neighbours
    and get inflated kernels.
    """
    tree = cKDTree(pos)
    # k = n_ngb + 1: the nearest 'neighbour' at distance 0 is the particle itself.
    dist, _ = tree.query(pos, k=[n_ngb + 1], workers=workers)
    return np.maximum(dist[:, 0], h_floor_mpc)


def cached_smoothing_lengths(cache_path, pos, n_ngb, h_floor_mpc, workers=1):
    """
    smoothing_lengths(), read from cache_path if a matching file is there and
    written to it otherwise.

    The particle order of `pos` must be the same on every call for one cluster
    (the builds concatenate dm, gas, star, bh in that order). A cached file
    whose particle count, n_ngb or floor disagrees is recomputed rather than
    trusted. Written to a temporary name and renamed, so a killed job never
    leaves a truncated cache behind.
    """
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path) as c:
                if (int(c["n_particles"]) == pos.shape[0]
                        and int(c["n_ngb"]) == int(n_ngb)
                        and float(c["h_floor_mpc"]) == float(h_floor_mpc)):
                    return c["h"].astype(np.float64)
        except (OSError, KeyError, ValueError):
            pass                                  # unreadable: recompute

    h = smoothing_lengths(pos, n_ngb, h_floor_mpc, workers=workers)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = f"{cache_path}.tmp{os.getpid()}.npz"
    np.savez(tmp, h=h.astype(np.float32), n_particles=np.int64(pos.shape[0]),
             n_ngb=np.int64(n_ngb), h_floor_mpc=np.float64(h_floor_mpc))
    os.replace(tmp, cache_path)
    # float32 on disk; return what a cache hit would, so the two paths agree.
    return h.astype(np.float32).astype(np.float64)


# ============================================================
# Deposit
# ============================================================

@nb.njit(inline="always", cache=True)
def _spline(q):
    """Gadget cubic spline in q = r / h, up to normalisation (renormalised per particle)."""
    if q < 0.5:
        return 1.0 - 6.0 * q * q + 6.0 * q * q * q
    if q < 1.0:
        t = 1.0 - q
        return 2.0 * t * t * t
    return 0.0


@nb.njit(parallel=True, cache=True)
def _deposit_spline(pos, m, h, N, L, sub, nth):
    """Mass per voxel (Msun, zyx) and the mass falling outside [-L, L]^3."""
    dx = 2.0 * L / N
    grids = np.zeros((nth, N, N, N))
    outside = np.zeros(nth)
    n = pos.shape[0]
    for t in nb.prange(nth):
        g = grids[t]
        for j in range(t, n, nth):
            x, y, z, hj = pos[j, 0], pos[j, 1], pos[j, 2], h[j]
            if (x + hj < -L or x - hj > L or y + hj < -L or y - hj > L
                    or z + hj < -L or z - hj > L):
                outside[t] += m[j]
                continue
            # Support entirely inside one voxel: the voxel integral is the
            # whole mass, exactly what the sub-sampling below would give, at a
            # fraction of the cost. Common for the small kernels of dense cores.
            vx0 = int(math.floor((x - hj + L) / dx)); vy0 = int(math.floor((y - hj + L) / dx))
            vz0 = int(math.floor((z - hj + L) / dx))
            if (vx0 == int(math.floor((x + hj + L) / dx))
                    and vy0 == int(math.floor((y + hj + L) / dx))
                    and vz0 == int(math.floor((z + hj + L) / dx))):
                if 0 <= vx0 < N and 0 <= vy0 < N and 0 <= vz0 < N:
                    g[vz0, vy0, vx0] += m[j]
                else:
                    outside[t] += m[j]
                continue
            s = max(1, int(math.ceil(sub * dx / hj)))
            d = dx / s
            i0 = int(math.floor((x - hj + L) / d)); i1 = int(math.floor((x + hj + L) / d))
            j0 = int(math.floor((y - hj + L) / d)); j1 = int(math.floor((y + hj + L) / d))
            k0 = int(math.floor((z - hj + L) / d)); k1 = int(math.floor((z + hj + L) / d))
            ih2 = 1.0 / (hj * hj)
            wsum = 0.0
            for a in range(i0, i1 + 1):
                ddx = -L + (a + 0.5) * d - x
                for b in range(j0, j1 + 1):
                    ddy = -L + (b + 0.5) * d - y
                    for c in range(k0, k1 + 1):
                        ddz = -L + (c + 0.5) * d - z
                        q2 = (ddx * ddx + ddy * ddy + ddz * ddz) * ih2
                        if q2 < 1.0:
                            wsum += _spline(math.sqrt(q2))
            if wsum <= 0.0:                       # unresolved: whole mass at the centre
                vx = int(math.floor((x + L) / dx)); vy = int(math.floor((y + L) / dx))
                vz = int(math.floor((z + L) / dx))
                if 0 <= vx < N and 0 <= vy < N and 0 <= vz < N:
                    g[vz, vy, vx] += m[j]
                else:
                    outside[t] += m[j]
                continue
            norm = m[j] / wsum
            for a in range(i0, i1 + 1):
                ddx = -L + (a + 0.5) * d - x
                vx = a // s
                for b in range(j0, j1 + 1):
                    ddy = -L + (b + 0.5) * d - y
                    vy = b // s
                    for c in range(k0, k1 + 1):
                        ddz = -L + (c + 0.5) * d - z
                        q2 = (ddx * ddx + ddy * ddy + ddz * ddz) * ih2
                        if q2 < 1.0:
                            w = norm * _spline(math.sqrt(q2))
                            vz = c // s
                            if 0 <= vx < N and 0 <= vy < N and 0 <= vz < N:
                                g[vz, vy, vx] += w
                            else:
                                outside[t] += w
    return grids.sum(axis=0), outside.sum()


def sph_density_cube(pos, mass, h, fov_mpc, N, sub=6.0, n_threads=1):
    """
    Voxel-averaged SPH density on an N^3 grid over [-fov_mpc, fov_mpc]^3.

    pos (n,3) Mpc in the viewing frame, mass (n,) Msun, h (n,) Mpc from
    smoothing_lengths. Returns (rho (N,N,N) float64 Msun/Mpc^3 in zyx order,
    mass outside the cube in Msun). n_threads = 1 inside multiprocessing
    workers, so numba's threads do not oversubscribe the pool.
    """
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    m_vox, outside = _deposit_spline(pos, mass, h, int(N), float(fov_mpc),
                                     float(sub), int(n_threads))
    voxel = (2.0 * fov_mpc / N) ** 3
    return m_vox / voxel, float(outside)


def block_average(rho, N_out):
    """
    Average an (N,N,N) voxel-averaged density down to (N_out,N_out,N_out).
    Exact for a voxel-averaged field: the mean of the children is the integral
    over the parent divided by its volume. N must be a multiple of N_out.
    """
    N = rho.shape[0]
    if N % N_out:
        raise ValueError(f"cannot block-average {N}^3 to {N_out}^3")
    f = N // N_out
    return rho.reshape(N_out, f, N_out, f, N_out, f).mean(axis=(1, 3, 5))


def multires_cubes(rho_fine, resolutions):
    """{N: block_average(rho_fine, N)} for every N in resolutions."""
    N_fine = rho_fine.shape[0]
    return {int(N): (rho_fine if int(N) == N_fine else block_average(rho_fine, int(N)))
            for N in resolutions}
