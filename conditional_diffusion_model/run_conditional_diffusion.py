import os
import sys
import argparse
import pickle
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from conditional_diffusion_3d_model import ConditionalUNet3D, DiffusionModelConfig
from train_conditional_diffusion import load_datasets, split_by_cluster, train_model

DATA_DIR = "/projects/mccleary_group/habjan.e/TNG/Data/shape_dynamics/"
# Pooled for training and validation; TNG300 is held out as the test set.
DEFAULT_FILES = ("bahamas_dataset.hdf5", "flamingo_dataset.hdf5")
RESOLUTIONS = (16, 32, 64, 128)


def parse_args():
    p = argparse.ArgumentParser(description="Train the conditional 3D diffusion model at one cube resolution.")
    p.add_argument("--resolution", type=int, required=True, choices=RESOLUTIONS,
                   help="N of the N^3 density cube (and the N x N conditioning images)")
    p.add_argument("--files", nargs="+", default=list(DEFAULT_FILES),
                   help=f"dataset files to pool, relative to {DATA_DIR} unless absolute")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="fraction of each suite's clusters held out for validation")
    p.add_argument("--split-seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    N = args.resolution
    files = [f if os.path.isabs(f) else os.path.join(DATA_DIR, f) for f in args.files]

    data = load_datasets(files, resolution=N)
    train_idx, val_idx = split_by_cluster(data, val_frac=args.val_frac, seed=args.split_seed)

    meta = data["metadata"]

    # Length of the per-source-galaxy embedding produced by the shape MLP
    # (the "C" of the latent 128x128x(C+1) shear cube).
    src_embed_dim = 32

    cfg = DiffusionModelConfig(
        base_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        out_channels=1,
        galaxy_token_dim=128,
        num_attention_heads=4,
        # gal_pixel_coords_N are on the N x N image grid.
        coord_image_size=N,
        src_embed_dim=src_embed_dim,
        src_out_ch=32,
        shape_grid_size=int(meta["shape_grid_size"]),
        # Mock-Chandra branch. map_fov_mpc sets the physical grid the angular
        # maps are reprojected onto, so it has to match the grid the cubes and
        # images were built on.
        xray_out_ch=32,
        xray_grid_size=int(meta["xray_grid_size"]),
        xray_reproj_size=128,
        map_fov_mpc=float(meta["map_fov_mpc"]),
    )

    print(f"X-ray branch: {cfg.xray_grid_size}x{cfg.xray_grid_size} angular maps "
          f"(counts_z, xray_mask, expmap) reprojected onto "
          f"{cfg.xray_reproj_size}x{cfg.xray_reproj_size} over +/-{cfg.map_fov_mpc} "
          f"Mpc, then strided to {N}x{N}x{cfg.xray_out_ch}")

    model = ConditionalUNet3D(cfg=cfg)

    batch_size = 2
    num_train_steps = 100_000
    eval_every = 250
    log_every = 50
    num_eval_batches = 25           # per validation suite

    lr = optax.cosine_decay_schedule(
        init_value=1e-4,
        decay_steps=int(num_train_steps * 0.9),
        alpha=0.1,
    )
    grad_clip = 1.0

    num_diffusion_steps = 1000
    beta_start = 1e-4
    beta_end = 2e-2

    suites = "+".join(meta["suites"])
    suffix = f"_shape_dyn_xray_{suites.lower().replace('+', '_')}_N{N}"

    wandb_notes = (
        f"Conditional 3D diffusion model on the shape_dynamics datasets, "
        f"{suites} pooled and split 80/20 by (suite, cluster). "
        f"{N}^3 adaptive-SPH cube (nominal log10 z-score 11/1) with {N}x{N} "
        f"dynamics images (gal_xy, gal_z_xy, gal_z_disp_xy). "
        "Weak-lensing source-galaxy point cloud replaces the projected mass "
        "map: per-source MLP embedding, mean-pooled per 128x128 cell, "
        f"+log1p(count) channel, stride-2 convs to {N}x{N}, concatenated onto "
        "the conditioning images. Galaxy token cross attention unchanged. "
        "Uniform MSE loss (no floor weighting)."
    )
    wandb_notes += (
        " Adds the mock-Chandra branch: the 128x128 maps on a fixed "
        "*angular* grid (20' ACIS-I, 0.7-2.6 Mpc across) are reprojected "
        "exactly (area-weighted, separable overlap matrices) onto a "
        "physical 128x128 grid over +/-fov, then strided down with convs. "
        "Pixels carry three states: detected (stored counts_z >= -1), "
        "observed with no detection (counts_z = -2), unobserved = off the "
        "detector footprint (counts_z = -5, exposure = -1); channels "
        "counts_z, exposure/104 cm^2, observed, observed fraction, and "
        "z_lens, log10 t_exp_ks, log10 mpc_per_pix broadcast."
    )

    cfg_dict = dict(
        base_channels=int(cfg.base_channels),
        channel_mults=str(cfg.channel_mults),
        time_emb_dim=int(cfg.time_emb_dim),
        out_channels=int(cfg.out_channels),
        galaxy_token_dim=int(cfg.galaxy_token_dim),
        num_attention_heads=int(cfg.num_attention_heads),
        coord_image_size=int(cfg.coord_image_size),
        src_embed_dim=int(cfg.src_embed_dim),
        src_out_ch=int(cfg.src_out_ch),
        shape_grid_size=int(cfg.shape_grid_size),
        image_storage="(S,C,H,W) in HDF5; transposed to (B,H,W,C) in loader",
        cube_storage="(S,Z,Y,X) in HDF5; expanded to (B,Z,Y,X,1) in loader",
        gal_feature_columns="(x, y, z_gal, Ngal)",
        gal_pixel_coord_columns="(x_pix, y_pix)",
        src_feature_columns="(e1, e2, ix, iy, z_s, z_lens, cell_frac)",
        resolution=int(N),
        data_files=",".join(files),
        suites=suites,
        split="by (suite, cluster_index), each suite separately",
        val_frac=float(args.val_frac),
        split_seed=int(args.split_seed),
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        cube_log10_mean=float(meta["cube_log10_mean"]),
        cube_log10_std=float(meta["cube_log10_std"]),
        cube_floor_value=float(meta["cube_floor_value"]),
        xray_out_ch=int(cfg.xray_out_ch),
        xray_grid_size=int(cfg.xray_grid_size),
        xray_reproj_size=int(cfg.xray_reproj_size),
        xray_nodet_counts_z=float(cfg.xray_nodet_counts_z),
        xray_unobs_counts_z=float(cfg.xray_unobs_counts_z),
        xray_unobs_exposure=float(cfg.xray_unobs_exposure),
        xray_exposure_ref_cm2=float(cfg.xray_exposure_ref_cm2),
        map_fov_mpc=float(cfg.map_fov_mpc),
        xray_map_channels="(counts_z, xray_mask, expmap)",
        xray_scalar_columns="(z_lens, log10 t_exp_ks, log10 xray_mpc_per_pix)",
    )

    trained_state, train_losses, test_losses, diffusion_cfg = train_model(
        data=data,
        train_idx=train_idx,
        val_idx=val_idx,
        model=model,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        eval_every=eval_every,
        log_every=log_every,
        num_eval_batches=num_eval_batches,
        learning_rate=lr,
        grad_clipping=grad_clip,
        num_diffusion_steps=num_diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        wandb_run_name=f"{suites}_N{N}",
        wandb_notes=wandb_notes,
        cfg_dict=cfg_dict,
        checkpoint_dir = "/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/cond_diff_old_models/checkpoints",
        checkpoint_prefix = f"cond_diffusion_runtime{suffix}",
        max_runtime_hours = 7.7,
        runtime_buffer_minutes = 15.0,
    )

    out_dir = os.path.join(HERE, "conditional_diffusion_models")
    os.makedirs(out_dir, exist_ok=True)

    param_path = os.path.join(out_dir, f"cond_diffusion_params{suffix}.pkl")
    sched_path = os.path.join(out_dir, f"cond_diffusion_schedule{suffix}.pkl")

    with open(param_path, "wb") as f:
        pickle.dump(trained_state.params, f)

    with open(sched_path, "wb") as f:
        pickle.dump(diffusion_cfg, f)

    loss_dir = "/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/Loss_arrays/"
    os.makedirs(loss_dir, exist_ok=True)

    train_loss_path = os.path.join(loss_dir, f"cond_diffusion_train_loss{suffix}.npy")
    test_loss_path = os.path.join(loss_dir, f"cond_diffusion_test_loss{suffix}.npy")

    np.save(train_loss_path, train_losses)
    np.save(test_loss_path, test_losses)

    print("Saved:")
    print(" ", param_path)
    print(" ", sched_path)
    print(" ", train_loss_path)
    print(" ", test_loss_path)


if __name__ == "__main__":
    main()