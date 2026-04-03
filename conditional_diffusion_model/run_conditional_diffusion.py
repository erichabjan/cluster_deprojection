import os
import sys
import pickle
import numpy as np
import optax

sys.path.append(os.getcwd())

from conditional_diffusion_3d_model import ConditionalUNet3D, DiffusionModelConfig
from train_conditional_diffusion import preload_hdf5_to_memory, train_model


def main():
    data_path = "/projects/mccleary_group/habjan.e/TNG/Data/conditional_diffusion_data/"
    train_file = "cond_diffusion_16cubed_train.h5"
    test_file = "cond_diffusion_16cubed_test.h5"

    train_path = os.path.join(data_path, train_file)
    test_path = os.path.join(data_path, test_file)

    train_data = preload_hdf5_to_memory(train_path)
    test_data = preload_hdf5_to_memory(test_path)

    cfg = DiffusionModelConfig(
        base_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        out_channels=1,
        galaxy_token_dim=128,
        num_attention_heads=4,
        coord_image_size=16,
    )

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

    suffix = "_16cube_v7"

    wandb_notes = (
        "Conditional 3D diffusion model"
        "Galaxy token cross attention and image conditioning"
        "Equal Fusion"
    )

    cfg_dict = dict(
        base_channels=int(cfg.base_channels),
        channel_mults=str(cfg.channel_mults),
        time_emb_dim=int(cfg.time_emb_dim),
        out_channels=int(cfg.out_channels),
        galaxy_token_dim=int(cfg.galaxy_token_dim),
        num_attention_heads=int(cfg.num_attention_heads),
        coord_image_size=int(cfg.coord_image_size),
        image_storage="(S,C,H,W) in HDF5; transposed to (B,H,W,C) in loader",
        cube_storage="(S,Z,Y,X) in HDF5; expanded to (B,Z,Y,X,1) in loader",
        gal_feature_columns="(x, y, vz, Ngal)",
        gal_pixel_coord_columns="(x_pix, y_pix)",
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