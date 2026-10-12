"""
Load a FLAMINGO cluster cutout in the conventions of the BAHAMAS npz files.

The FLAMINGO cutouts (/projects/mccleary_group/habjan.e/FLAMINGO/<run>/
cluster_<idx>.npz) carry their own unit table and differ from BAHAMAS in both
units and field names. Rather than teach xray_utils and the dataset build a
second convention, load_cluster converts on load into exactly what a BAHAMAS
GrNm_*.npz holds, so every downstream (a/h) factor, density_unit_cgs and the
box wrapping run unchanged and the final hdf5 comes out in BAHAMAS units:

    quantity         FLAMINGO cutout         returned (BAHAMAS)
    *_pos, CoP, R200 Mpc                     comoving Mpc/h
    *_mass, M200     Msun                    Msun (h already divided out)
    *_vel            km/s                    km/s
    gas_temp         K                       K
    gas_dens         Msun/Mpc^3              Gadget code density,
                                             1e10 Msun/h per (cMpc/h)^3
    BoxSize          Mpc                     comoving Mpc/h, under
                                             config.BOXSIZE_KEY

Subhalos are the HBT-HERONS catalogue (hbt_*), mapped onto BAHAMAS's
SUBFIND-style sub_* fields. FLAMINGO's massType columns are (gas, dm, star,
bh); BAHAMAS's sub_massTotal is indexed by Gadget particle type
(0 gas, 1 dm, 4 star, 5 bh), so the columns are scattered into a (N, 6) array
with types 2 and 3 left at zero.

Only the gas fields BAHAMAS has are returned (pos, vel, mass, temp, dens), so
the X-ray photon lists are built from exactly the same inputs in both suites.
Neutrino particles are dropped, matching BAHAMAS, which has none.

Positions are comoving in the snapshots while SOAP radii are physical; the two
only coincide at a = 1, and every cutout here is at z = 0. Rather than guess at
the convention of each field, load_cluster refuses a != 1.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOXSIZE_KEY

# Units load_cluster expects to find in the cutout's own unit table. A cutout
# written differently fails here instead of being silently mis-scaled.
_EXPECTED_UNITS = {
    "BoxSize": "Mpc", "CoP": "Mpc", "R200": "Mpc", "M200": "Msun",
    "host_Mfof": "Msun", "host_r500c": "Mpc", "host_m500c": "Msun",
    "gas_pos": "Mpc", "gas_vel": "km/s", "gas_mass": "Msun",
    "gas_temp": "K", "gas_dens": "Msun/Mpc**3",
    "dm_pos": "Mpc", "dm_vel": "km/s", "dm_mass": "Msun",
    "star_pos": "Mpc", "star_vel": "km/s", "star_mass": "Msun",
    "bh_pos": "Mpc", "bh_vel": "km/s", "bh_mass": "Msun",
    "hbt_pos": "Mpc", "hbt_vel": "km/s", "hbt_massType": "Msun",
    "hbt_mass30kpc": "Msun",
}

# FLAMINGO massType column -> BAHAMAS (Gadget) particle-type column.
_MASSTYPE_TO_GADGET = {0: 0, 1: 1, 2: 4, 3: 5}       # gas, dm, star, bh

_SPECIES = ("gas", "dm", "star", "bh")


class ClusterData(dict):
    """
    A dict with an npz-style `.files`, so code written against np.load's
    NpzFile (`key in data.files`, `data[key]`) takes it unchanged.
    """

    @property
    def files(self):
        return list(self.keys())


def _check_units(raw, path):
    table = dict(zip((str(k) for k in raw["units_fields"]),
                     (str(v) for v in raw["units_values"])))
    bad = []
    for key, want in _EXPECTED_UNITS.items():
        if key not in raw.files:
            continue                        # e.g. no stars in a run without them
        have = table.get(key)
        if have != want:
            bad.append(f"{key}: expected {want!r}, file says {have!r}")
    if bad:
        raise ValueError(f"{path}: unexpected units\n  " + "\n  ".join(bad))


def _to_gadget_columns(mass_type):
    out = np.zeros((mass_type.shape[0], 6), dtype=np.float64)
    for src, dst in _MASSTYPE_TO_GADGET.items():
        out[:, dst] = mass_type[:, src]
    return out


def load_cluster(npz_path):
    """
    One FLAMINGO cutout as a ClusterData holding the BAHAMAS npz keys, in
    BAHAMAS units (see the module docstring), plus:

      config.BOXSIZE_KEY   periodic box, comoving Mpc/h
      R500c_soap           SOAP's R500c of the host, comoving Mpc/h
      M500c_soap           SOAP's M500c of the host, Msun
      simulation, cluster_index, halo_index   bookkeeping, as stored

    The two SOAP values are cross-checks for xray_utils.r500c_mpc_over_h; the
    pipeline itself solves R500c from the particles, as it does for BAHAMAS.
    """
    raw = np.load(npz_path)
    _check_units(raw, npz_path)

    h = float(raw["h"])
    a = float(raw["a"])
    if abs(a - 1.0) > 1e-6:
        raise ValueError(
            f"{npz_path}: a = {a}. load_cluster only handles z = 0 cutouts, "
            f"where comoving and physical lengths coincide.")

    # Mpc -> comoving Mpc/h (a = 1, so comoving = physical).
    length = h
    out = ClusterData()
    out["h"] = np.float64(h)
    out["a"] = np.float64(a)
    out["CoP"] = np.asarray(raw["CoP"], dtype=np.float64) * length
    out["R200"] = np.float64(float(raw["R200"]) * length)
    out["M200"] = np.float64(raw["M200"])
    out["Mfof"] = np.float64(raw["host_Mfof"])
    out[BOXSIZE_KEY] = np.float64(float(raw["BoxSize"]) * length)

    for sp in _SPECIES:
        if f"{sp}_pos" not in raw.files:
            continue
        out[f"{sp}_pos"] = np.asarray(raw[f"{sp}_pos"], dtype=np.float64) * length
        out[f"{sp}_vel"] = np.asarray(raw[f"{sp}_vel"], dtype=np.float64)
        out[f"{sp}_mass"] = np.asarray(raw[f"{sp}_mass"], dtype=np.float64)

    out["gas_temp"] = np.asarray(raw["gas_temp"], dtype=np.float64)
    # Msun/Mpc^3 (physical) -> 1e10 Msun/h per (cMpc/h)^3, the Gadget code
    # density BAHAMAS stores; xray_utils.density_unit_cgs inverts exactly this:
    #   code = rho * a^3 / (1e10 * h^2).
    out["gas_dens"] = (np.asarray(raw["gas_dens"], dtype=np.float64)
                       * a ** 3 / (1.0e10 * h ** 2))

    # HBT-HERONS subhalos in place of BAHAMAS's SUBFIND sub_* fields.
    out["sub_pos"] = np.asarray(raw["hbt_pos"], dtype=np.float64) * length
    out["sub_vel"] = np.asarray(raw["hbt_vel"], dtype=np.float64)
    out["sub_massTotal"] = _to_gadget_columns(np.asarray(raw["hbt_massType"]))
    out["sub_mass30kpc"] = _to_gadget_columns(np.asarray(raw["hbt_mass30kpc"]))

    out["R500c_soap"] = np.float64(float(raw["host_r500c"]) * length)
    out["M500c_soap"] = np.float64(raw["host_m500c"])
    out["simulation"] = str(raw["simulation"])
    out["cluster_index"] = int(raw["cluster_index"])
    out["halo_index"] = int(raw["halo_index"])
    return out
