"""
Load a TNG300 cluster in the conventions of the BAHAMAS npz files.

The TNG300 inputs are the per-particle-type 5 R200 cutouts in
<TNG300_DATA_ROOT>/5r200_data-{1,2,3}/{dm,gas,stars,bh}_within_5r200_<halo>.hdf5
(made by TNG_cluster_dynamics/rockstar/tng_cutout_scripts) plus the SUBFIND
group/subhalo catalogs. load_cluster returns exactly what a BAHAMAS
GrNm_*.npz holds, so xray_utils and the dataset build run unchanged and the
final hdf5 comes out in BAHAMAS units, like flamingo_io does for FLAMINGO:

    quantity          TNG                          returned (BAHAMAS)
    *_pos, CoP, R200  ckpc/h                       comoving Mpc/h      (/ 1e3)
    *_mass, M200,     1e10 Msun/h                  Msun                (* 1e10 / h)
      Mfof
    dm_mass           not stored per particle      header MassTable[1], per resolution
    *_vel             km/s * sqrt(a) (particles),  km/s                (a = 1)
                      km/s (subhalos)
    gas_dens          1e10 Msun/h / (ckpc/h)^3     Gadget code density,
                                                   1e10 Msun/h / (cMpc/h)^3 (* 1e9)
    gas_temp          from InternalEnergy and      K
                      ElectronAbundance
    sub_massTotal     SubhaloMassType              Msun, same Gadget column order
                                                   (0 gas, 1 dm, 4 stars, 5 bh)

Members: every SUBFIND subhalo within the cutout radius (5 R200) of the group
centre, whichever FoF group it belongs to -- the same sphere as FLAMINGO.

Stars: PartType4 wind particles (GFM_StellarFormationTime <= 0) are gas in
transit, not stars, and are dropped.

Only the gas fields BAHAMAS has are returned (pos, vel, mass, temp, dens), so
the X-ray photon lists are built from the same inputs in all three suites.

Fields are read lazily, on first access: a TNG300-1 cluster carries ~10^8
particles, and the pipeline never touches some fields (DM velocities, for
one), so loading everything up front would cost several GB per worker.
"""
import os
import sys

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOXSIZE_KEY, TNG300_DATA_ROOT, TNG300_COSMOLOGY

SNAP_A = 1.0                         # snapshot 99, z = 0 (header Time = 1 - 2e-16)
BOXSIZE_CKPC_H = 205000.0

CUTOUT_DIR = {"TNG300-1": "5r200_data-1", "TNG300-2": "5r200_data-2",
              "TNG300-3": "5r200_data-3"}
# DM particle mass, header MassTable[1] of each resolution, 1e10 Msun/h.
DM_PARTICLE_MASS = {"TNG300-1": 0.003983427498675485,
                    "TNG300-2": 0.03186741998940388,
                    "TNG300-3": 0.25493935991523103}

# Group/subhalo catalogs (iapi_TNG single-field files, Group/<f> or
# Subhalo/<f>), searched in this order. TNG300-1's full GroupMass catalog
# cannot be fetched from the API (504s), so its Mfof comes from the per-halo
# top-100 npz instead.
CATALOG_DIRS = ("/home/habjan.e/TNG/Data/TNG_data", TNG300_DATA_ROOT.rstrip("/"))
GROUPMASS_TOP100 = {"TNG300-1": os.path.join(TNG300_DATA_ROOT,
                                             "TNG300-1_GroupMass_top100.npz")}

# Gas thermodynamics, for T from the internal energy (the standard TNG
# conversion): mu = 4 / (1 + 3 X_H + 4 X_H x_e) m_p, T = (gamma - 1) u mu / k_B.
X_H = 0.76
GAMMA = 5.0 / 3.0
M_P_G = 1.67262192369e-24
K_B_ERG_K = 1.380649e-16
UNIT_U_CGS = 1.0e10                  # (km/s)^2 -> (cm/s)^2

_SPECIES = {"dm": 1, "gas": 0, "star": 4, "bh": 5}
_CUTOUT_PTYPE = {"dm": "dm", "gas": "gas", "star": "stars", "bh": "bh"}

# Subhalo catalogs, loaded once per process and shared by every cluster of a
# resolution: TNG300-1 has 14.5M subhalos.
_SUBHALO_CACHE = {}


def _catalog_path(sim, field):
    for d in CATALOG_DIRS:
        p = os.path.join(d, f"{sim}_{field}.hdf5")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no {sim}_{field}.hdf5 in {CATALOG_DIRS}")


def _group_value(sim, field, halo_id):
    with h5py.File(_catalog_path(sim, field), "r") as f:
        return np.asarray(f["Group"][field][halo_id])


def _group_mass(sim, halo_id):
    """Mfof (GroupMass), 1e10 Msun/h."""
    try:
        return float(_group_value(sim, "GroupMass", halo_id))
    except FileNotFoundError:
        if sim not in GROUPMASS_TOP100:
            raise
        d = np.load(GROUPMASS_TOP100[sim])
        if halo_id >= d["GroupMass"].size:
            raise IndexError(f"{GROUPMASS_TOP100[sim]} only covers halos "
                             f"0-{d['GroupMass'].size - 1}")
        return float(d["GroupMass"][halo_id])


def _subhalos(sim):
    if sim not in _SUBHALO_CACHE:
        out = {}
        for field in ("SubhaloPos", "SubhaloVel", "SubhaloMassType"):
            with h5py.File(_catalog_path(sim, field), "r") as f:
                out[field] = np.asarray(f["Subhalo"][field])
        _SUBHALO_CACHE[sim] = out
    return _SUBHALO_CACHE[sim]


def cutout_path(sim, species, halo_id):
    return os.path.join(TNG300_DATA_ROOT, CUTOUT_DIR[sim],
                        f"{_CUTOUT_PTYPE[species]}_within_5r200_{halo_id}.hdf5")


class ClusterData(dict):
    """
    A dict with an npz-style `.files`, so code written against np.load's
    NpzFile (`key in data.files`, `data[key]`) takes it unchanged. Values that
    are callables are loaders, run on first access and then cached.
    """

    @property
    def files(self):
        return list(self.keys())

    def __getitem__(self, key):
        v = dict.__getitem__(self, key)
        if callable(v):
            v = v()
            dict.__setitem__(self, key, v)
        return v

    def get(self, key, default=None):
        return self[key] if key in self else default


def _check_cutout(path, species, sim, halo_id, rcut_ckpc_h):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} (run the {species} cutout for {sim} halo {halo_id})")
    with h5py.File(path, "r") as f:
        if species == "gas" and f.attrs.get("xray_fields_added") is None:
            raise RuntimeError(f"{path} has no X-ray gas fields yet "
                               f"(run run_tng_5r200_cutouts_gas.slurm)")
        rc = float(f.attrs["rcut"])
    if abs(rc - rcut_ckpc_h) > 1e-6 * rcut_ckpc_h:
        raise ValueError(f"{path}: rcut {rc} ckpc/h, expected {rcut_ckpc_h}")


def _reader(path, ptype, field, scale=1.0, mask=None):
    """Loader for one cutout dataset, as float64 times `scale`."""
    def load():
        with h5py.File(path, "r") as f:
            v = np.asarray(f[f"PartType{ptype}"][field][:], dtype=np.float64)
        if mask is not None:
            v = v[mask()]
        if scale != 1.0:
            v *= scale
        return v
    return load


def _gas_temperature(path):
    def load():
        with h5py.File(path, "r") as f:
            g = f["PartType0"]
            u = np.asarray(g["InternalEnergy"][:], dtype=np.float64) * UNIT_U_CGS
            x_e = np.asarray(g["ElectronAbundance"][:], dtype=np.float64)
        mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * x_e)
        return (GAMMA - 1.0) * u * mu * M_P_G / K_B_ERG_K
    return load


def load_cluster(sim, halo_id):
    """
    TNG300 halo `halo_id` of resolution `sim` as a ClusterData holding the
    BAHAMAS npz keys in BAHAMAS units (see the module docstring), plus
    config.BOXSIZE_KEY (comoving Mpc/h), simulation and cluster_index.

    Raises if any of the four cutouts is missing or the gas cutout lacks the
    X-ray fields, rather than building a sample from part of the matter.
    """
    if sim not in CUTOUT_DIR:
        raise KeyError(f"{sim!r} is not a TNG300 resolution ({sorted(CUTOUT_DIR)})")
    h, _ = TNG300_COSMOLOGY[sim]
    a = SNAP_A
    length = 1.0e-3                        # ckpc/h -> cMpc/h
    mass = 1.0e10 / h                      # 1e10 Msun/h -> Msun
    halo_id = int(halo_id)

    center_ckpc = _group_value(sim, "GroupPos", halo_id).astype(np.float64)
    r200_ckpc = float(_group_value(sim, "Group_R_Crit200", halo_id))
    m200 = float(_group_value(sim, "Group_M_Crit200", halo_id))
    rcut_ckpc = 5.0 * r200_ckpc
    paths = {s: cutout_path(sim, s, halo_id) for s in _SPECIES}
    for s, p in paths.items():
        _check_cutout(p, s, sim, halo_id, rcut_ckpc)

    out = ClusterData()
    out["h"] = np.float64(h)
    out["a"] = np.float64(a)
    out["CoP"] = center_ckpc * length
    out["R200"] = np.float64(r200_ckpc * length)
    out["M200"] = np.float64(m200 * mass)
    out["Mfof"] = np.float64(_group_mass(sim, halo_id) * mass)
    out[BOXSIZE_KEY] = np.float64(BOXSIZE_CKPC_H * length)
    out["simulation"] = str(sim)
    out["cluster_index"] = halo_id

    # Particles. Velocities: km/s * sqrt(a) in the snapshot, i.e. km/s at a = 1.
    vel_scale = np.sqrt(a)
    for s, pt in _SPECIES.items():
        p = paths[s]
        mask = None
        if s == "star":
            def mask(p=p):
                with h5py.File(p, "r") as f:
                    return np.asarray(f["PartType4"]["GFM_StellarFormationTime"][:]) > 0
            mask = _memo(mask)
        out[f"{s}_pos"] = _reader(p, pt, "Coordinates", length, mask)
        out[f"{s}_vel"] = _reader(p, pt, "Velocities", vel_scale, mask)
        if s == "dm":
            out["dm_mass"] = _dm_mass(p, DM_PARTICLE_MASS[sim] * mass)
        else:
            out[f"{s}_mass"] = _reader(p, pt, "Masses", mass, mask)

    # Gas thermodynamics. (1e10 Msun/h)/(ckpc/h)^3 -> (1e10 Msun/h)/(cMpc/h)^3.
    out["gas_dens"] = _reader(paths["gas"], 0, "Density", 1.0e9)
    out["gas_temp"] = _gas_temperature(paths["gas"])

    # SUBFIND subhalos within the cutout radius of the group centre.
    sub = _subhalos(sim)
    d = sub["SubhaloPos"].astype(np.float64) - center_ckpc
    d = (d + 0.5 * BOXSIZE_CKPC_H) % BOXSIZE_CKPC_H - 0.5 * BOXSIZE_CKPC_H
    inside = np.flatnonzero(np.einsum("ij,ij->i", d, d) <= rcut_ckpc ** 2)
    out["sub_index"] = inside
    out["sub_pos"] = sub["SubhaloPos"][inside].astype(np.float64) * length
    out["sub_vel"] = sub["SubhaloVel"][inside].astype(np.float64)     # km/s, peculiar
    out["sub_massTotal"] = sub["SubhaloMassType"][inside].astype(np.float64) * mass
    return out


def _memo(fn):
    cache = []

    def wrapped():
        if not cache:
            cache.append(fn())
        return cache[0]
    return wrapped


def _dm_mass(path, m_msun):
    def load():
        with h5py.File(path, "r") as f:
            n = f["PartType1"]["Coordinates"].shape[0]
        return np.full(n, m_msun, dtype=np.float64)
    return load
