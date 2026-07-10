#!/usr/bin/env python3
"""
Prep step (run once, before build_shape_dynamics_dataset.py): build the
LoVoCCS empirical source-redshift template used by the shape_dynamics
pipeline (CFG_LENS.empirical_template = 'LoVoCCS').

Streams the five LoVoCCS gen-2 cluster catalogs and collects every finite BPZ
point-estimate redshift (column 'z_b') into a single n(z) template array,
saved as {OUT_NPZ}[{OUT_KEY}]. A3825 has no full *_bpz_merge.csv, so its
cut/shear-calibrated catalog is used instead.

Run: cl_dyn env, single CPU (make_lovoccs_template.slurm). Re-run only if the
LoVoCCS catalogs change.
"""
import csv
import math
import os
import numpy as np

LOVOCCS_DIR = "/projects/mccleary_group/habjan.e/LoVoCCS/LoVoCCS_gen_2"
CSV_FILES = [
    "A2384_00-1111_gal_dered_dezp_bpz_merge.csv",
    "A3571_00-1111_gal_dered_dezp_bpz_merge.csv",
    "A3667_00-1111_gal_dered_dezp_bpz_merge.csv",
    "A3827_00-1111_gal_dered_dezp_bpz_merge.csv",
    # A3825 has no full *_bpz_merge.csv; use the cut catalog.
    "A3825_00-1111_gal_dered_dezp_bpz_merge_cut_shear_calib_merge_cut.csv",
]
Z_COLUMN = "z_b"

OUT_NPZ = "/home/habjan.e/TNG/Data/lovoccs_redshifts/redshift_arrays.npz"
OUT_KEY = "bpz_redshift"


def extract_z(csv_path, z_column):
    """Stream one catalog and return its finite z_column values."""
    vals = []
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        try:
            zi = header.index(z_column)
        except ValueError:
            raise ValueError(f"{csv_path}: no column {z_column!r} in header")
        n_bad = 0
        for row in reader:
            try:
                z = float(row[zi])
            except (ValueError, IndexError):
                n_bad += 1
                continue
            if math.isfinite(z):
                vals.append(z)
            else:
                n_bad += 1
    return np.asarray(vals, dtype=np.float64), n_bad


def main():
    per_cluster = {}
    for fname in CSV_FILES:
        path = os.path.join(LOVOCCS_DIR, fname)
        z, n_bad = extract_z(path, Z_COLUMN)
        cluster = fname.split("_")[0]
        per_cluster[cluster] = z
        print(f"{cluster}: {z.size} finite {Z_COLUMN} "
              f"(skipped {n_bad} bad/non-finite), "
              f"mean={z.mean():.3f}, median={np.median(z):.3f}, "
              f"min={z.min():.3f}, max={z.max():.3f}", flush=True)

    z_all = np.concatenate(list(per_cluster.values()))
    print(f"combined: {z_all.size} redshifts, mean={z_all.mean():.3f}, "
          f"median={np.median(z_all):.3f}, min={z_all.min():.3f}, max={z_all.max():.3f}")

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    payload = {OUT_KEY: z_all}
    # Per-cluster arrays kept alongside for diagnostics.
    payload.update({f"{OUT_KEY}_{c}": z for c, z in per_cluster.items()})
    np.savez_compressed(OUT_NPZ, **payload)
    print(f"saved {OUT_NPZ} [{OUT_KEY}]")


if __name__ == "__main__":
    main()
