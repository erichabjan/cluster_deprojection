#!/usr/bin/env python3
"""
Stage 1: pyXSIM photon lists for one FLAMINGO cluster.

The FLAMINGO counterpart of BAHAMAS/xray_photons.py, and deliberately the same
pipeline: the cutout is converted to BAHAMAS units and field names on load
(flamingo_io.load_cluster), and the photon list is then made by the same
xray_utils calls from the same gas fields (pos, vel, mass, temp, dens), with
the same source model (X_H, Z_met, T and n_H cuts). Only the per-run cosmology
differs, taken from config.sim_cosmology.

Nothing here depends on the viewing direction: a photon list is a 3D sample of
the cluster's X-ray emission, and it is only turned into a sky image later, by
project_photons in build_flamingo_dataset.py. It does depend on redshift, so
one list is written per *sample*, at that sample's exact z_lens from
config.sample_z_lens -- CFG_DATA.n_projections = 1 for FLAMINGO, so one list
per cluster.

The cluster is centred on its BCG, matching the rest of the pipeline.

Usage (see xray_photons.slurm):
    python xray_photons.py <cluster_id> <sim>        e.g. 000 L1_m9
"""
import os
import sys
import time
import argparse

# Select the FLAMINGO configuration before config is first imported.
os.environ["CLUSTER_SUITE"] = "FLAMINGO"

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG_XRAY, CFG_DATA, sample_z_lens, sim_cosmology
import xray_utils as xu
import flamingo_io


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cluster_id", help="cluster index, e.g. 000")
    p.add_argument("sim", help="FLAMINGO run directory, e.g. L1_m9")
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

    # The lists themselves go to /projects, one subdirectory per run.
    # exist_ok because 28 array tasks race to create the same directory.
    os.makedirs(xu.photon_sim_dir(sim), exist_ok=True)

    # pyXSIM writes a few products relative to the working directory, and the
    # array runs 28 of these at once, so each task gets its own scratch cwd.
    task_dir = os.path.join(CFG_XRAY.photon_work_dir, f"{sim}_{cluster_idx:03d}")
    os.makedirs(task_dir, exist_ok=True)
    os.chdir(task_dir)

    n_proj = CFG_DATA.n_projections
    if args.max_proj:
        n_proj = min(n_proj, args.max_proj)
    # One list per sample, at that sample's own z_lens. Same keying as the
    # dataset build, so the two stages cannot disagree about the redshift.
    plan = [(p, sample_z_lens(sim, cluster_idx, p)) for p in range(n_proj)]

    npz_path = os.path.join(CFG_DATA.sim_data_root, sim, f"{stem}_{cluster_idx:03d}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    todo = [(p, z) for p, z in plan
            if args.overwrite
            or not xu.photon_list_is_complete(xu.photon_prefix(sim, cluster_idx, p))]
    print(f"{name}: {len(todo)} of {len(plan)} projections "
          f"to build -> {xu.photon_sim_dir(sim)}", flush=True)
    print("  z_lens: " + ", ".join(f"p{p}={z:.5f}" for p, z in plan), flush=True)
    if not todo:
        return

    # BAHAMAS units and keys from here on (positions cMpc/h, gas_dens in
    # Gadget code units, HBT-HERONS subhalos as sub_*).
    data = flamingo_io.load_cluster(npz_path)
    h = float(data["h"])
    h_cfg, Om0 = sim_cosmology(sim)
    if abs(h - h_cfg) > 1e-4:
        raise ValueError(f"{name}: cutout has h = {h}, config.FLAMINGO_COSMOLOGY "
                         f"has {h_cfg}; one of them is for the wrong run")

    bcg = xu.bcg_index(data)
    center = data["sub_pos"][bcg]
    cop_offset = np.linalg.norm(center - data["CoP"])
    print(f"  BCG = HBT subhalo {bcg}, M* = {data['sub_massTotal'][bcg, 4]:.3e} Msun, "
          f"{cop_offset:.4f} cMpc/h from the CoP; "
          f"M200 = {float(data['M200']):.3e} Msun, R200 = {float(data['R200']):.3f} cMpc/h; "
          f"h = {h}, Om0 = {Om0}", flush=True)

    # Hot gas within n_R200 * R200 of the BCG, as an in-memory yt SPH dataset.
    ds, radius_mpc = xu.load_gas_dataset(data, center)
    cosmo = xu.bahamas_cosmology(h, Om0)
    n_hot = int(ds.all_data()["hot_gas", "particle_mass"].size)
    print(f"  {n_hot:,} hot gas particles within {CFG_XRAY.n_R200} R200 "
          f"({radius_mpc:.2f} pMpc)", flush=True)
    if n_hot == 0:
        raise RuntimeError(
            f"{name}: no gas passes the hot-gas filter "
            f"(T > {CFG_XRAY.T_min_K:.0e} K and n_H < {CFG_XRAY.nH_max} cm^-3) "
            f"within {CFG_XRAY.n_R200} R200. pyXSIM cannot sample photons from "
            f"an empty source; check flamingo_io's gas_temp/gas_dens conversion."
        )

    # LoVoCCS's X-ray selection is L_X500c(0.1-2.4 keV) > 1e44 erg/s. Reported
    # here as a diagnostic; nothing is ever dropped on it. SOAP's R500c is
    # printed alongside the particle solve as a cross-check.
    lx, r500, n_lx = xu.lx500c_erg_s(data, center)
    print(f"  L_X500c({CFG_XRAY.lx_band_kev[0]}-{CFG_XRAY.lx_band_kev[1]} keV) "
          f"= {lx:.3e} erg/s from {n_lx:,} particles inside R500c = {r500:.3f} "
          f"cMpc/h (SOAP: {float(data['R500c_soap']):.3f})  [LoVoCCS cut: 1e44]",
          flush=True)

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

    print(f"Done: {name}")


if __name__ == "__main__":
    main()
