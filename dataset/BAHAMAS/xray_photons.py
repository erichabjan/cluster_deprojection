#!/usr/bin/env python3
"""
Stage 1: pyXSIM photon lists for one BAHAMAS cluster.

Ported from Sandbox_notebooks/mock_xray_maps/init_xray_map_code.ipynb, up to and
including make_photons. Nothing here depends on the viewing direction: a photon
list is a 3D sample of the cluster's X-ray emission, and it is only turned into
a sky image later, by project_photons in build_bahamas_dataset.py.

It does depend on redshift, which sets both the flux and the angular scale, so
one list is written per *sample* -- CFG_DATA.n_projections of them per cluster,
each at that sample's exact z_lens from config.sample_z_lens. That is cheaper
than the redshift binning it replaces (5 lists per cluster against the 9 bins a
0.01-wide binning of [0.03, 0.12] needed) and it removes the up-to-14%
angular-scale mismatch a bin centre left between a sample's X-ray map and its
lensing geometry.

config.sample_z_lens is the contract between this script and the dataset build:
it is keyed on (simulation, cluster, projection) rather than on a sequential rng
stream, so both stages agree without either having to replay the other's draws.

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
from config import CFG_XRAY, CFG_DATA, sample_z_lens
import xray_utils as xu


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cluster_id", help="cluster index, e.g. 001")
    p.add_argument("sim", help="simulation directory, e.g. CDMb")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate photon lists that already exist")
    p.add_argument("--max-proj", type=int, default=0,
                   help="only the first N projections (0 = all); for smoke tests")
    return p.parse_args()


def main():
    args = parse_args()
    cluster_idx = int(args.cluster_id)
    sim = args.sim

    # The lists themselves go to /projects, one subdirectory per dark-matter
    # model. exist_ok because 28 array tasks race to create the same directory.
    os.makedirs(xu.photon_sim_dir(sim), exist_ok=True)

    # pyXSIM writes a few products relative to the working directory, and the
    # array runs 28 of these at once. Give each task its own scratch directory
    # so no two can ever collide over a fixed-name intermediate. This stays on
    # /scratch: it is regenerable, and it should not litter the output tree.
    # The photon lists themselves are written to absolute paths under photon_dir.
    task_dir = os.path.join(CFG_XRAY.photon_work_dir, f"{sim}_{cluster_idx:03d}")
    os.makedirs(task_dir, exist_ok=True)
    os.chdir(task_dir)

    n_proj = CFG_DATA.n_projections
    if args.max_proj:
        n_proj = min(n_proj, args.max_proj)
    # One list per sample, at that sample's own z_lens. Same keying as the
    # dataset build, so the two stages cannot disagree about which redshift
    # projection p of this cluster is observed at.
    plan = [(p, sample_z_lens(sim, cluster_idx, p)) for p in range(n_proj)]

    npz_path = os.path.join(CFG_DATA.sim_data_root, sim, f"GrNm_{cluster_idx:03d}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    todo = [(p, z) for p, z in plan
            if args.overwrite
            or not xu.photon_list_is_complete(xu.photon_prefix(sim, cluster_idx, p))]
    print(f"{sim} GrNm_{cluster_idx:03d}: {len(todo)} of {len(plan)} projections "
          f"to build -> {xu.photon_sim_dir(sim)}", flush=True)
    print("  z_lens: " + ", ".join(f"p{p}={z:.5f}" for p, z in plan), flush=True)
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
    if n_hot == 0:
        raise RuntimeError(
            f"{sim} GrNm_{cluster_idx:03d}: no gas passes the hot-gas filter "
            f"(T > {CFG_XRAY.T_min_K:.0e} K and n_H < {CFG_XRAY.nH_max} cm^-3) "
            f"within {CFG_XRAY.n_R200} R200. pyXSIM cannot sample photons from "
            f"an empty source; check the npz's gas_temp/gas_dens units."
        )

    # LoVoCCS's X-ray selection is L_X500c(0.1-2.4 keV) > 1e44 erg/s. Reported
    # here as a diagnostic; nothing is ever dropped on it.
    lx, r500, n_lx = xu.lx500c_erg_s(data, center)
    print(f"  L_X500c({CFG_XRAY.lx_band_kev[0]}-{CFG_XRAY.lx_band_kev[1]} keV) "
          f"= {lx:.3e} erg/s from {n_lx:,} particles inside R500c = {r500:.3f} "
          f"cMpc/h  [LoVoCCS cut: 1e44]", flush=True)

    for p, z in todo:
        prefix = xu.photon_prefix(sim, cluster_idx, p)
        t0 = time.time()
        n_ph, n_cells = xu.make_photon_list(prefix, ds, radius_mpc, z, cosmo)
        size_gb = os.path.getsize(prefix + ".h5") / 1024 ** 3
        print(f"  p{p} z = {z:.5f}: {n_ph:,} photons from {n_cells:,} particles, "
              f"{size_gb:.2f} GB, {time.time() - t0:.0f} s", flush=True)

    try:
        os.chdir(CFG_XRAY.photon_work_dir)
        os.rmdir(task_dir)          # only succeeds if pyXSIM left nothing behind
    except OSError:
        pass

    print(f"Done: {sim} GrNm_{cluster_idx:03d}")


if __name__ == "__main__":
    main()
