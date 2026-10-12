import os
import sys
import pickle
import numpy as np
import optax

sys.path.append(os.getcwd())

from conditional_diffusion_3d_model import ConditionalUNet3D, DiffusionModelConfig
from train_conditional_diffusion import preload_hdf5_to_memory, train_model


def main():
    data_path = "/projects/mccleary_group/habjan.e/TNG/Data/shape_dynamics/"
    train_file = "shape_dynamics_train.h5"
    test_file = "shape_dynamics_val.h5"

    train_path = os.path.join(data_path, train_file)
    test_path = os.path.join(data_path, test_file)

    train_data = preload_hdf5_to_memory(train_path)
    test_data = preload_hdf5_to_memory(test_path)

    # The X-ray branch is driven entirely by what the loader found, so a file
    # predating the standardized maps trains without it. Report that rather
    # than letting it pass silently -- the model's xray_maps argument defaults
    # to None, so the run would look normal either way.
    has_xray = "xray_maps" in train_data
    if has_xray != ("xray_maps" in test_data):
        raise ValueError(
            "train and val disagree on X-ray data "
            f"(train={has_xray}, val={'xray_maps' in test_data}); the batch "
            "shapes would differ between train_step and eval_step"
        )

    meta = train_data["metadata"]

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
        coord_image_size=16,
        src_embed_dim=src_embed_dim,
        src_out_ch=32,
        shape_grid_size=int(meta["shape_grid_size"]),
        # Mock-Chandra branch. map_fov_mpc is what converts the sample's
        # angular pixel scale into a conditioning-cell index, so it has to
        # match the grid the cubes and images were built on.
        xray_embed_dim=32,
        xray_out_ch=32,
        xray_grid_size=int(meta.get("xray_grid_size", 128)),
        xray_sentinel_z=float(meta.get("xray_sentinel_z", -5.0)),
        map_fov_mpc=float(meta.get("map_fov_mpc", 5.0)),
    )

    if has_xray:
        print(f"X-ray branch ON: {cfg.xray_grid_size}x{cfg.xray_grid_size} maps "
              f"(mosaic_z, counts_z, xray_mask, density_mask) -> {cfg.xray_out_ch} "
              f"channels scattered onto the {cfg.coord_image_size}x"
              f"{cfg.coord_image_size} grid over +/-{cfg.map_fov_mpc} Mpc")
    else:
        print(f"X-ray branch OFF: no xray_mask in {train_file}. The maps are "
              f"only in the standardized builds; this run conditions on "
              f"dynamics + lensing alone.")

    model = ConditionalUNet3D(cfg=cfg)

    batch_size = 2
    num_train_steps = 100_000
    eval_every = 250
    log_every = 50
    num_eval_batches = 50

    lr = optax.cosine_decay_schedule(
        init_value=1e-4,
        decay_steps=int(num_train_steps * 0.9),
        alpha=0.1,
    )
    grad_clip = 1.0

    num_diffusion_steps = 1000
    beta_start = 1e-4
    beta_end = 2e-2

    suffix = "_shape_dyn_xray_v1" if has_xray else "_shape_dyn_v1"

    wandb_notes = (
        "Conditional 3D diffusion model on the shape_dynamics dataset. "
        "16^3 cube with 16x16 dynamics images (gal_xy, vz, vz_disp). "
        "Weak-lensing source-galaxy point cloud replaces the projected mass "
        "map: per-source MLP embedding, mean-pooled per 128x128 cell, "
        "+log1p(count) channel, stride-2 convs to 16x16, concatenated onto "
        "the conditioning images. Galaxy token cross attention unchanged. "
        "Uniform MSE loss (no floor weighting)."
    )
    if has_xray:
        wandb_notes += (
            " Adds the mock-Chandra branch: the 128x128 maps are on a fixed "
            "*angular* grid (20' ACIS-I), so unlike the shear branch they "
            "cannot be strided down -- the field is only 0.7-2.6 Mpc across "
            "depending on z_lens. Instead a per-pixel MLP over (mosaic_z, "
            "counts_z, xray_mask, density_mask, z_lens, log10 t_exp_ks, "
            "log10 mpc_per_pix) is mean-pooled into whichever conditioning "
            "cell the pixel physically lands in, with log1p(n_detected) and "
            "detected-fraction channels. Cells the field never reaches take a "
            "learned sentinel embedding from the same MLP rather than zeros. "
            "At 16x16 the field spans only 1-6 cells, so the branch acts as a "
            "mass/dynamical-state proxy."
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
        xray_branch=bool(has_xray),
        xray_embed_dim=int(cfg.xray_embed_dim),
        xray_out_ch=int(cfg.xray_out_ch),
        xray_grid_size=int(cfg.xray_grid_size),
        xray_sentinel_z=float(cfg.xray_sentinel_z),
        map_fov_mpc=float(cfg.map_fov_mpc),
        xray_map_channels="(mosaic_z, counts_z, xray_mask, density_mask)",
        xray_scalar_columns="(z_lens, log10 t_exp_ks, log10 xray_mpc_per_pix)",
    )

    trained_state, train_losses, test_losses, diffusion_cfg = train_model(
        train_data=train_data,
        test_data=test_data,
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
        wandb_notes=wandb_notes,
        cfg_dict=cfg_dict,
        checkpoint_dir = "/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/cond_diff_old_models/checkpoints",
        checkpoint_prefix = f"cond_diffusion_runtime{suffix}",
        max_runtime_hours = 7.7,
        runtime_buffer_minutes = 15.0,
    )

    out_dir = os.path.join(os.getcwd(), "conditional_diffusion_models")
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