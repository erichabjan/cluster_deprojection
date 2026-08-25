#!/usr/bin/env python3
"""
Stage 1: pyXSIM photon lists for one BAHAMAS cluster.

Ported from Sandbox_notebooks/mock_xray_maps/init_xray_map_code.ipynb, up to and
including make_photons. Nothing here depends on the viewing direction: a photon
list is a 3D sample of the cluster's X-ray emission, and it is only turned into
a sky image later, by project_photons in build_bahamas_dataset.py. So the cost
of the expensive step is paid once per cluster instead of once per projection.

It does depend on redshift, which sets both the flux and the angular scale, so
one list is written per redshift bin across [z_lens_min, z_lens_max]. Every
sample then uses the list within half a bin of its own z_lens.

The cluster is centred on its BCG, matching the rest of the pipeline.

Usage (see xray_photons.slurm):
    python xray_photons.py <cluster_id> <sim>        e.g. 001 CDMb
"""
import os
import sys
import time
import argparse

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG_XRAY, CFG_DATA
import xray_utils as xu


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cluster_id", help="cluster index, e.g. 001")
    p.add_argument("sim", help="simulation directory, e.g. CDMb")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate photon lists that already exist")
    p.add_argument("--max-bins", type=int, default=0,
                   help="only the first N redshift bins (0 = all); for smoke tests")
    return p.parse_args()


def main():
    args = parse_args()
    cluster_idx = int(args.cluster_id)
    sim = args.sim

    os.makedirs(CFG_XRAY.photon_dir, exist_ok=True)
    # pyXSIM writes a few products relative to the working directory.
    os.chdir(CFG_XRAY.photon_dir)

    z_bins = xu.z_bin_centers()
    if args.max_bins:
        z_bins = z_bins[:args.max_bins]

    npz_path = os.path.join(CFG_DATA.sim_data_root, sim, f"GrNm_{cluster_idx:03d}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    todo = [z for z in z_bins
            if args.overwrite
            or not os.path.exists(xu.photon_prefix(sim, cluster_idx, z) + ".h5")]
    print(f"{sim} GrNm_{cluster_idx:03d}: {len(todo)} of {len(z_bins)} redshift "
          f"bins to build -> {CFG_XRAY.photon_dir}", flush=True)
    if not todo:
        return

    data = np.load(npz_path)
    h = float(data["h"])
    bcg = xu.bcg_index(data)
    center = data["sub_pos"][bcg]
    cop_offset = np.linalg.norm(center - data["CoP"])
    print(f"  BCG = subhalo {bcg}, M* = {data['sub_massTotal'][bcg, 4]:.3e} Msun, "
          f"{cop_offset:.4f} cMpc/h from the CoP; "
          f"M200 = {float(data['M200']):.3e} Msun, R200 = {float(data['R200']):.3f} cMpc/h",
          flush=True)

    # Hot gas within n_R200 * R200 of the BCG, as an in-memory yt SPH dataset.
    ds, radius_mpc = xu.load_gas_dataset(data, center)
    cosmo = xu.bahamas_cosmology(h)
    n_hot = int(ds.all_data()["hot_gas", "particle_mass"].size)
    print(f"  {n_hot:,} hot gas particles within {CFG_XRAY.n_R200} R200 "
          f"({radius_mpc:.2f} pMpc)", flush=True)

    for z in todo:
        prefix = xu.photon_prefix(sim, cluster_idx, z)
        t0 = time.time()
        n_ph, n_cells = xu.make_photon_list(prefix, ds, radius_mpc, z, cosmo)
        size_gb = os.path.getsize(prefix + ".h5") / 1024 ** 3
        print(f"  z = {z:.4f}: {n_ph:,} photons from {n_cells:,} particles, "
              f"{size_gb:.2f} GB, {time.time() - t0:.0f} s", flush=True)

    print(f"Done: {sim} GrNm_{cluster_idx:03d}")


if __name__ == "__main__":
    main()
