#!/usr/bin/env python3
"""
Stage 1: pyXSIM photon lists for one TNG300 halo.

The TNG300 counterpart of BAHAMAS/ and FLAMINGO/xray_photons.py, and
deliberately the same pipeline: the cutouts are converted to BAHAMAS units and
field names on load (tng300_io.load_cluster), and the photon lists are made
by the same xray_utils calls from the same gas fields (pos, vel, mass, temp,
dens), with the same source model (X_H, Z_met, T and n_H cuts, 5.7 kpc
smoothing floor).

Nothing here depends on the viewing direction: a photon list is a 3D sample of
the cluster's X-ray emission, and it is only turned into a sky image later, by
project_photons in build_tng300_dataset.py. It does depend on redshift, so one
list is written per *sample*, at that sample's exact z_lens from
config.sample_z_lens. A halo has config.n_projections_for(sim, halo) samples
-- 8, or 9 for the 100 halos that bring the suite to 2500 -- and the dataset
build calls the same function, so the two stages agree on the sample list.

The cluster is centred on its BCG, matching the rest of the pipeline.

Usage (see xray_photons.slurm):
    python xray_photons.py <halo_id> <sim>        e.g. 000 TNG300-1
"""
import os
import sys
import time
import argparse

# Select the TNG300 configuration before config is first imported.
os.environ["CLUSTER_SUITE"] = "TNG300"

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG_XRAY, CFG_DATA, sample_z_lens, sim_cosmology, n_projections_for
import xray_utils as xu
import tng300_io


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cluster_id", help="halo index, e.g. 000")
    p.add_argument("sim", help="TNG300 resolution, e.g. TNG300-1")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate photon lists that already exist")
    p.add_argument("--max-proj", type=int, default=0,
                   help="only the first N projections (0 = all); for smoke tests")
    return p.parse_args()


def main():
    args = parse_args()
    cluster_idx = int(args.cluster_id)
    sim = args.sim
    stem = CFG_DATA.cluster_file_stem
    name = f"{sim} {stem}_{cluster_idx:03d}"

    # The lists themselves go to /projects, one subdirectory per resolution.
    # exist_ok because the array tasks race to create the same directory.
    os.makedirs(xu.photon_sim_dir(sim), exist_ok=True)

    # pyXSIM writes a few products relative to the working directory, and the
    # array runs many of these at once, so each task gets its own scratch cwd.
    task_dir = os.path.join(CFG_XRAY.photon_work_dir, f"{sim}_{cluster_idx:03d}")
    os.makedirs(task_dir, exist_ok=True)
    os.chdir(task_dir)

    n_proj = n_projections_for(sim, cluster_idx)
    if args.max_proj:
        n_proj = min(n_proj, args.max_proj)
    # One list per sample, at that sample's own z_lens. Same keying as the
    # dataset build, so the two stages cannot disagree about the redshift.
    plan = [(p, sample_z_lens(sim, cluster_idx, p)) for p in range(n_proj)]

    todo = [(p, z) for p, z in plan
            if args.overwrite
            or not xu.photon_list_is_complete(xu.photon_prefix(sim, cluster_idx, p))]
    print(f"{name}: {len(todo)} of {len(plan)} projections "
          f"to build -> {xu.photon_sim_dir(sim)}", flush=True)
    print("  z_lens: " + ", ".join(f"p{p}={z:.5f}" for p, z in plan), flush=True)
    if not todo:
        return

    # BAHAMAS units and keys from here on (positions cMpc/h, gas_dens in
    # Gadget code units, SUBFIND subhalos within 5 R200 as sub_*).
    t_load = time.time()
    data = tng300_io.load_cluster(sim, cluster_idx)
    h = float(data["h"])
    _, Om0 = sim_cosmology(sim)

    bcg = xu.bcg_index(data)
    center = data["sub_pos"][bcg]
    cop_offset = np.linalg.norm(center - data["CoP"])
    print(f"  BCG = subhalo {int(data['sub_index'][bcg])}, "
          f"M* = {data['sub_massTotal'][bcg, 4]:.3e} Msun, "
          f"{cop_offset:.4f} cMpc/h from the CoP; "
          f"M200 = {float(data['M200']):.3e} Msun, R200 = {float(data['R200']):.3f} cMpc/h; "
          f"h = {h}, Om0 = {Om0}", flush=True)

    # Hot gas within n_R200 * R200 of the BCG, as an in-memory yt SPH dataset.
    ds, radius_mpc = xu.load_gas_dataset(data, center)
    cosmo = xu.bahamas_cosmology(h, Om0)
    n_hot = int(ds.all_data()["hot_gas", "particle_mass"].size)
    print(f"  {n_hot:,} hot gas cells within {CFG_XRAY.n_R200} R200 "
          f"({radius_mpc:.2f} pMpc), loaded in {time.time() - t_load:.0f} s", flush=True)
    if n_hot == 0:
        raise RuntimeError(
            f"{name}: no gas passes the hot-gas filter "
            f"(T > {CFG_XRAY.T_min_K:.0e} K and n_H < {CFG_XRAY.nH_max} cm^-3) "
            f"within {CFG_XRAY.n_R200} R200. pyXSIM cannot sample photons from "
            f"an empty source; check tng300_io's gas_temp/gas_dens conversion."
        )

    # LoVoCCS's X-ray selection is L_X500c(0.1-2.4 keV) > 1e44 erg/s. Reported
    # here as a diagnostic; nothing is ever dropped on it.
    lx, r500, n_lx = xu.lx500c_erg_s(data, center)
    print(f"  L_X500c({CFG_XRAY.lx_band_kev[0]}-{CFG_XRAY.lx_band_kev[1]} keV) "
          f"= {lx:.3e} erg/s from {n_lx:,} cells inside R500c = {r500:.3f} "
          f"cMpc/h  [LoVoCCS cut: 1e44]", flush=True)

    for p, z in todo:
        prefix = xu.photon_prefix(sim, cluster_idx, p)
        t0 = time.time()
        n_ph, n_cells = xu.make_photon_list(prefix, ds, radius_mpc, z, cosmo)
        size_gb = os.path.getsize(prefix + ".h5") / 1024 ** 3
        print(f"  p{p} z = {z:.5f}: {n_ph:,} photons from {n_cells:,} cells, "
              f"{size_gb:.2f} GB, {time.time() - t0:.0f} s", flush=True)

    try:
        os.chdir(CFG_XRAY.photon_work_dir)
        os.rmdir(task_dir)          # only succeeds if pyXSIM left nothing behind
    except OSError:
        pass

    print(f"Done: {name}")


if __name__ == "__main__":
    main()
