import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
import h5py
import time

import os
import sys
import pickle

sys.path.append(os.getcwd())
from training_structure import train_model, data_loader, create_train_state, train_step, preload_hdf5_to_memory
from gnn import GraphConvNet

### Add a suffix for a new model
suffix = '_testing_newdata'

### Import data and create data loaders
data_path = "/projects/mccleary_group/habjan.e/TNG/Data/GNN_SBI_data/"
train_file = 'GNN_data_train.h5'
test_file = 'GNN_data_test.h5'

### Train model
if __name__ == "__main__":

    train_data = preload_hdf5_to_memory(data_path, train_file)
    test_data = preload_hdf5_to_memory(data_path, test_file)

    ### Model paramters
    hidden_size = 1024
    num_mlp_layers = 3
    latent_size = 128
 
    # Create and train the model
    model = GraphConvNet(latent_size = latent_size, 
                         hidden_size = hidden_size, 
                         num_mlp_layers = num_mlp_layers, 
                         message_passing_steps = 5, 
                         skip_connections = True,
                         edge_skip_connections = True,
                         norm = "none", 
                         attention = True,
                         shared_weights = True,
                         relative_updates = False,
                         output_dim = 2,
                         dropout_rate = 0.0)


    ### Training parameters
    batch_size = 4

    early_stopping = False
    patience = 50
    num_train_steps = 50_000
    eval_every = 25
    log_every = 50
    num_eval_batches = 100

    learning_rate = learning_rate = optax.cosine_decay_schedule(init_value=1e-3,decay_steps=int(num_train_steps * 0.9), alpha=0.1)
    gradient_clipping = 1

    ### Weights and Biases Notes
    wandb_notes = (
        "Learning rate decay run. "
        f"hidden_size={hidden_size}, num_mlp_layers={num_mlp_layers}. "
        f"learning_rate={learning_rate}. "
        "Same as baseline run, except shared_weights = True, no early stoppinf=g, fixed velocities"
    )

    trained_state, model, train_losses, test_losses = train_model(
        train_data=train_data,
        test_data=test_data,
        model=model, 
        batch_size = batch_size,
        num_train_steps = num_train_steps, 
        eval_every = eval_every, 
        log_every = log_every,
        num_eval_batches=num_eval_batches,
        learning_rate=learning_rate,
        grad_clipping=gradient_clipping,
        latent_size=latent_size,
        early_stopping=early_stopping, 
        patience=patience,
        wandb_notes = wandb_notes,
        hidden_size = hidden_size, 
        num_mlp_layers = num_mlp_layers,
    )

    # Save model parameters
    save_path = os.getcwd() + '/GNN_models/gnn_model_params' + suffix + '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(trained_state.params, f)
    
    save_data = '/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/Loss_arrays/'
    np.save(save_data + 'train_loss' + suffix + '.npy', train_losses)
    np.save(save_data + 'test_loss' + suffix + '.npy', test_losses)