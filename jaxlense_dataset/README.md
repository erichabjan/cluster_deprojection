# jaxlense_dataset

Pipeline that produces a conditional-diffusion training/test dataset where the
projected-mass conditioning channel is replaced by a **realistic convergence
map** reconstructed with `jax-lensing`'s probabilistic mass-mapping
(DLPosterior, Remy+ 2021).

Per-(cluster, sim, projection) sample:

1. Build true convergence κ_∞ from the simulation matter components.
2. Draw a synthetic background-galaxy shape catalog (Bahé+ 2012 §3.2 with a
   Smail n(z), shape noise σ_ε=0.2, magnification thinning).
3. Bin observed shear γ̂_1, γ̂_2 from the catalog at the lensing-reconstruction
   resolution.
4. Train a UResNet18 score network on the cluster κ prior.
5. Posterior-sample κ_E given (γ̂_1, γ̂_2) with annealed-HMC DLPosterior.
6. Assemble the final HDF5 with 8 conditioning channels at any chosen final
   image/cube resolution.

```
sim NPZ  ──▶  Stage 1 (cl_dyn, CPU)  ──▶  intermediate/sample_*.npz
                                          {κ_∞, γ̂_1, γ̂_2, n_gal_pix, cube, members, …}

intermediate/  ──▶  Stage 2-prep (cl_dyn, CPU)  ──▶  score_weights/
                                                     {cluster_kappa_PS_theory.npy, mean_beta.npy, pixel_size_rad.npy}

intermediate/  +  score_weights/  ──▶  Stage 2a (jax_lense, GPU)  ──▶  score_weights/score_model-final.pckl

intermediate/  +  score_weights/  ──▶  Stage 2b (jax_lense, GPU)  ──▶  posterior/posterior_*.npz
                                                                       {κ_E_mean, κ_E_std} at lens_recon_resolution

intermediate/  +  posterior/  ──▶  Stage 3 (cl_dyn, CPU)  ──▶  final/cond_diffusion_jaxlense_{img}img_{cube}cubed_{train,test}.h5
```

## Files

### Shared libraries

| File | Role |
|---|---|
| `config.py` | Dataclasses for grid / lensing / cosmology / scaling / dataset / score-net / DLPosterior knobs. Production defaults; `JAXLENSE_SMOKE_TEST=1` shrinks to ~12 samples + tiny training/sampling for end-to-end smoke checks. |
| `lensing_utils.py` | Pure-numpy utilities: flat-ΛCDM distances, `Σ_crit(z_l, z_s=∞)`, `β(z_l, z_s)`, Smail n(z) sampling, `⟨β⟩`, KS93 forward/inverse (matches `jax_lensing.inversion`), bilinear sampling, 2D binning, avg-pool helpers. Importable from both `cl_dyn` and `jax_lense`. |

### Stage 1 — synthetic catalog + truth κ (cl_dyn, CPU)

| File | Role |
|---|---|
| `build_kappa_truth_and_catalogs.py` | For each (cluster, sim, projection): rotate matter into the viewing frame, build Σ→κ_∞ on a 128² × 10 Mpc grid, generate the §3.2 shape catalog (Smail n(z), inner mask 30″, outer mask 10′, magnification thinning), bin (γ̂_1, γ̂_2, n_gal_pix), bin a 64³ density cube, and snapshot the rotated cluster-member catalog. Writes one `intermediate/sample_NNNNNN.npz` per sample plus a global `manifest.npz` with the train/test split (held out by cluster). |
| `build_kappa_truth_and_catalogs.slurm` | CPU job spec (cl_dyn, 10 CPUs, 24 G, partition=short). |

### Stage 2-prep — Gaussian-prior P(k) and ⟨β⟩ (cl_dyn, CPU)

| File | Role |
|---|---|
| `compute_kappa_power_spectrum.py` | Average power spectrum of the training-set κ_train = ⟨β⟩·κ_∞ maps (the prior the score network learns). Saves `score_weights/cluster_kappa_PS_theory.npy` (jax-lensing's `(ell, P)` format), `mean_beta.npy`, `pixel_size_rad.npy`. |
| `compute_kappa_power_spectrum.slurm` | CPU job spec. |

### Stage 2a — score-network training (jax_lense, GPU)

| File | Role |
|---|---|
| `train_score_for_clusters.py` | Reads κ_train maps directly from intermediate npz files (custom numpy loader — no `tensorflow` import), trains `jax_lensing.models.UResNet18` as a denoising score network with the cluster Gaussian prior. Writes `score_weights/score_model-final.pckl`. |
| `train_score_for_clusters.slurm` | GPU job spec (jax_lense, 1× a100, 24 G, partition=gpu). |

### Stage 2b — DLPosterior κ sampling (jax_lense, GPU)

| File | Role |
|---|---|
| `run_dlposterior.py` | Loads stage-1 (γ̂_1, γ̂_2, n_gal_pix), the trained score-net weights, and the cluster Gaussian prior. Per sample, runs `tfp.mcmc.sample_chain` with `TemperedMC + ScoreHamiltonianMonteCarlo` (annealed HMC over the score-prior + masked-Gaussian-likelihood). Writes posterior mean + per-pixel std to `posterior/posterior_NNNNNN.npz`. Supports `--start/--stop` and `sbatch --array=…` for parallel chunks. |
| `run_dlposterior.slurm` | GPU job spec; auto-splits the manifest across array tasks when `--array=…` is used. |

### Stage 3 — final HDF5 assembly (cl_dyn, CPU)

| File | Role |
|---|---|
| `assemble_dataset.py` | Joins stage-1 + stage-2b outputs and writes the final train/test HDF5. The κ-related lensing channels are avg-pooled from 128² → `--final-image-resolution`; the density cube is avg-pooled from 64³ → `--final-cube-resolution`; the dynamical channels (gal_xy, gal_vz_xy, gal_vz_disp_xy) are re-binned at the chosen final resolution from the saved cluster-member catalog. n_gal_pix is summed (not averaged) to preserve count semantics. Writes the 8-channel image stack `[kappa_E_mean, kappa_E_std, gamma1_obs, gamma2_obs, n_gal_pix, gal_xy, gal_vz_xy, gal_vz_disp_xy]` per sample, plus the 3D cube target, padded galaxy catalog, and global mass/axis-length targets — all z-score normalized using train-set statistics. |
| `assemble_dataset.slurm` | CPU job spec. Defaults to 16²/16³; override with `FINAL_IMG_RES=32 FINAL_CUBE_RES=32 sbatch assemble_dataset.slurm`. |

### Smoke-test helper (not part of production runs)

| File | Role |
|---|---|
| `_smoketest_fake_posteriors.py` | Generates placeholder `posterior_*.npz` files (kappa_E_mean = numpy KS93 of γ̂; kappa_E_std = σ_ε/√N_pix) so Stage 3 can be exercised end-to-end without an actual DLPosterior pass. Use only with `JAXLENSE_SMOKE_TEST=1` for verifying the pipeline's data flow when the GPU stages are blocked. |

## Run order

```bash
cd /home/habjan.e/TNG/cluster_deprojection/jaxlense_dataset

# Smoke-test the whole pipeline on ~12 samples (set JAXLENSE_SMOKE_TEST=1 in env / slurm export)
sbatch --export=ALL,JAXLENSE_SMOKE_TEST=1 build_kappa_truth_and_catalogs.slurm
sbatch --export=ALL,JAXLENSE_SMOKE_TEST=1 compute_kappa_power_spectrum.slurm
sbatch --export=ALL,JAXLENSE_SMOKE_TEST=1 train_score_for_clusters.slurm
sbatch --export=ALL,JAXLENSE_SMOKE_TEST=1 --array=0-1 run_dlposterior.slurm
sbatch --export=ALL,JAXLENSE_SMOKE_TEST=1 assemble_dataset.slurm

# Production: drop the JAXLENSE_SMOKE_TEST var; defaults from config.py kick in.
sbatch build_kappa_truth_and_catalogs.slurm
sbatch compute_kappa_power_spectrum.slurm
sbatch train_score_for_clusters.slurm
sbatch --array=0-9 run_dlposterior.slurm
sbatch assemble_dataset.slurm
# Higher final resolution later — no rerun of GPU stages needed:
FINAL_IMG_RES=32 FINAL_CUBE_RES=32 sbatch assemble_dataset.slurm
```

## Output layout

```
$output_root = /projects/mccleary_group/habjan.e/TNG/Data/jaxlense_dataset/
  manifest.npz                              # train_ids, test_ids, n_total
  intermediate/sample_NNNNNN.npz            # one per (cluster, sim, projection)
  score_weights/cluster_kappa_PS_theory.npy
  score_weights/mean_beta.npy
  score_weights/pixel_size_rad.npy
  score_weights/score_model-final.pckl
  posterior/posterior_NNNNNN.npz            # one per sample, lens_recon_resolution
  final/cond_diffusion_jaxlense_{img}img_{cube}cubed_{train,test}.h5
```

## Status

- Stages 1, 2-prep, 3 verified end-to-end on the smoke-test config (`cl_dyn`).
- Stages 2a / 2b (GPU, `jax_lense`) currently blocked by a `dm-haiku 0.0.11` /
  `chex 0.1.90` / `jax 0.4.30` pytree-registry incompatibility (`FlatMap` vs
  plain `dict` mismatch on `tree_map`). The forward pass and gradients work;
  the failure is at the optimizer-update boundary. Resolution path is open
  (likely either a flax rewrite of `UResNet18` or a curated downgrade of
  `chex` / `dm-haiku`).
