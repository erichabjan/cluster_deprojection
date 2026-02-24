import os
import sys
import pickle
import numpy as np
import optax

import jax

sys.path.append(os.getcwd())
from cnn_mlp_model import CNNMLPModel, ModelConfig
from train_cnn_mlp import preload_hdf5_to_memory, train_model


def main():

    data_path = "/projects/mccleary_group/habjan.e/TNG/Data/CNN_MLP_data/"
    train_file = "CNN_MLP_train.h5"
    test_file = "CNN_MLP_test.h5"

    train_path = os.path.join(data_path, train_file)
    test_path = os.path.join(data_path, test_file)

    train_data = preload_hdf5_to_memory(train_path)
    test_data = preload_hdf5_to_memory(test_path)

    cfg = ModelConfig(
        smoother_kernel=5,

        fmap_channels=(32, 64, 128),
        fmap_blocks_per_stage=2,
        cnn_dropout=0.0,

        mlp_hidden=(128, 128),
        head_hidden=(256, 256),
        dropout=0.0,

        output_dim=1,
    )
    model = CNNMLPModel(cfg=cfg)

    batch_size = 16
    num_train_steps = 50_000
    eval_every = 250
    log_every = 50
    num_eval_batches = 250

    lr = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=int(num_train_steps * 0.9), alpha=0.1,)
    grad_clip = 10.0

    suffix = "_cnnmlp_sampler_v1"

    wandb_notes = (
        "Residual feature-map encoder + bilinear sampler. "
        "Model inputs: images (mass_xy, gal_xy, gal_vz_xy), gal_features=(x,y,vz,Ngal), "
        "gal_pixel_coords=(x_pix,y_pix). "
        "Sampler extracts local CNN features at each galaxy position, concatenated with MLP branch."
    )

    cfg_dict = dict(
        smoother_kernel=cfg.smoother_kernel,
        fmap_channels=str(cfg.fmap_channels),
        fmap_blocks_per_stage=int(cfg.fmap_blocks_per_stage),
        cnn_dropout=float(cfg.cnn_dropout),
        mlp_hidden=str(cfg.mlp_hidden),
        head_hidden=str(cfg.head_hidden),
        dropout=float(cfg.dropout),
        output_dim=int(cfg.output_dim),
        image_storage="(S,C,H,W) in HDF5; transposed to (B,H,W,C) in loader",
        model_inputs="images, mlp_in, gal_pixel_coords",
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
    )

    out_dir = os.path.join(os.getcwd(), "CNN_MLP_models")
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"cnn_mlp_params{suffix}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(trained_state.params, f)

    loss_dir = "/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/Loss_arrays/"
    os.makedirs(loss_dir, exist_ok=True)
    np.save(os.path.join(loss_dir, f"train_loss{suffix}.npy"), train_losses)
    np.save(os.path.join(loss_dir, f"test_loss{suffix}.npy"), test_losses)

    print("Saved:")
    print(" ", save_path)
    print(" ", os.path.join(loss_dir, f"train_loss{suffix}.npy"))
    print(" ", os.path.join(loss_dir, f"test_loss{suffix}.npy"))


if __name__ == "__main__":
    main()