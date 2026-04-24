import os
import sys
import pickle
import numpy as np
import optax

sys.path.append(os.getcwd())

from cnn_mlp_globals_model import CNNSetGlobalsModel, GlobalsModelConfig
from train_cnn_mlp_globals import preload_hdf5_to_memory, train_model


def main():
    data_path = "/projects/mccleary_group/habjan.e/TNG/Data/conditional_diffusion_data/"
    train_file = "cond_diffusion_16cubed_16img_train.h5"
    test_file = "cond_diffusion_16cubed_16img_test.h5"

    train_path = os.path.join(data_path, train_file)
    test_path = os.path.join(data_path, test_file)

    train_data = preload_hdf5_to_memory(train_path)
    test_data = preload_hdf5_to_memory(test_path)

    cfg = GlobalsModelConfig(
        base_channels=32,
        channel_mults=(1, 2, 4),
        galaxy_token_dim=128,
        num_attention_heads=4,
        num_global_queries=8,
        coord_image_size=16,
        head_hidden=(256, 256),
        out_dim=4,
    )

    model = CNNSetGlobalsModel(cfg=cfg)

    batch_size = 32
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

    suffix = "_16img_v1"

    wandb_notes = (
        "CNN + galaxy-token + global-query attention pool. "
        "Point estimator for cube enclosed mass log10(Msun) and three "
        "shape-tensor axis lengths (Mpc), all standardized. "
        "MSE loss across 4 outputs, equally weighted."
    )

    cfg_dict = dict(
        base_channels=int(cfg.base_channels),
        channel_mults=str(cfg.channel_mults),
        galaxy_token_dim=int(cfg.galaxy_token_dim),
        num_attention_heads=int(cfg.num_attention_heads),
        num_global_queries=int(cfg.num_global_queries),
        coord_image_size=int(cfg.coord_image_size),
        head_hidden=str(cfg.head_hidden),
        out_dim=int(cfg.out_dim),
        target_columns="(mass_log10_msun, axis_a_mpc, axis_b_mpc, axis_c_mpc) standardized",
        image_storage="(S,C,H,W) in HDF5; transposed to (B,H,W,C) in loader",
    )

    trained_state, train_losses, test_losses = train_model(
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
        wandb_notes=wandb_notes,
        cfg_dict=cfg_dict,
        checkpoint_dir="/home/habjan.e/TNG/cluster_deprojection/cnn_mlp_globals/checkpoints",
        checkpoint_prefix=f"cnn_mlp_globals_runtime{suffix}",
        max_runtime_hours=7.7,
        runtime_buffer_minutes=15.0,
    )

    out_dir = os.path.join(os.getcwd(), "cnn_mlp_globals_models")
    os.makedirs(out_dir, exist_ok=True)

    param_path = os.path.join(out_dir, f"cnn_mlp_globals_params{suffix}.pkl")

    with open(param_path, "wb") as f:
        pickle.dump(trained_state.params, f)

    loss_dir = "/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/Loss_arrays/"
    os.makedirs(loss_dir, exist_ok=True)

    train_loss_path = os.path.join(loss_dir, f"cnn_mlp_globals_train_loss{suffix}.npy")
    test_loss_path = os.path.join(loss_dir, f"cnn_mlp_globals_test_loss{suffix}.npy")

    np.save(train_loss_path, train_losses)
    np.save(test_loss_path, test_losses)

    print("Saved:")
    print(" ", param_path)
    print(" ", train_loss_path)
    print(" ", test_loss_path)


if __name__ == "__main__":
    main()
