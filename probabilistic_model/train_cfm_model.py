import os
import sys
import pickle
import numpy as np

import jax

sys.path.append(os.getcwd())

from cfm_training_structure import (
    preload_hdf5_to_memory,
    train_model,
)
from cfm_gnn import GraphConvNet, CFMGraphModel


suffix = "_cfm_testing"

data_path = "/projects/mccleary_group/habjan.e/TNG/Data/GNN_SBI_data/"
train_file = "GNN_data_train.h5"
test_file = "GNN_data_test.h5"


if __name__ == "__main__":

    train_data = preload_hdf5_to_memory(data_path, train_file)
    test_data = preload_hdf5_to_memory(data_path, test_file)

    hidden_size = 1024
    num_mlp_layers = 3
    latent_size = 128
    target_dim = 3
    time_emb_dim = 128

    backbone = GraphConvNet(
        latent_size=latent_size,
        hidden_size=hidden_size,
        num_mlp_layers=num_mlp_layers,
        message_passing_steps=5,
        skip_connections=True,
        edge_skip_connections=True,
        norm="none",
        attention=True,
        shared_weights=True,
        relative_updates=False,
        output_dim=target_dim,
        dropout_rate=0.0,
    )

    model = CFMGraphModel(
        backbone=backbone,
        target_dim=target_dim,
        time_emb_dim=time_emb_dim,
    )

    batch_size = 4
    early_stopping = True
    patience = 200
    num_train_steps = 50_000
    eval_every = 25
    log_every = 50
    num_eval_batches = 100

    learning_rate = 3e-5
    gradient_clipping = 1.0

    wandb_notes = (
        "CFM run (linear-path flow matching). "
        f"hidden_size={hidden_size}, num_mlp_layers={num_mlp_layers}, latent_size={latent_size}. "
        f"time_emb_dim={time_emb_dim}."
        f"learning_rate={learning_rate}. "
        "Backbone: same ConvGNN; wrapper adds (x_t, t) conditioning."
    )

    trained_state, model, train_losses, test_losses = train_model(
        train_data=train_data,
        test_data=test_data,
        model=model,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        eval_every=eval_every,
        log_every=log_every,
        num_eval_batches=num_eval_batches,
        learning_rate=learning_rate,
        grad_clipping=gradient_clipping,
        early_stopping=early_stopping,
        patience=patience,
        wandb_notes=wandb_notes,
        hidden_size=hidden_size,
        num_mlp_layers=num_mlp_layers,
        target_dim=target_dim,
    )

    # Save model parameters
    save_path = os.path.join(os.getcwd(), "CFM_models", f"cfm_model_params{suffix}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(trained_state.params, f)

    save_data = "/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/Loss_arrays_cfm/"
    np.save(os.path.join(save_data, f"train_loss{suffix}.npy"), train_losses)
    np.save(os.path.join(save_data, f"test_loss{suffix}.npy"), test_losses)

    print(f"Saved params to: {save_path}")
