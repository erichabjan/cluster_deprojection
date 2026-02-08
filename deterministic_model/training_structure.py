import optax 
from flax.training import train_state

import jax
import jax.numpy as jnp
from jax import random
import time
import numpy as np
from functools import partial

from typing import Iterator, Tuple, Dict
import jraph

import time
import os
import h5py

from jax.tree_util import tree_leaves

import random
import wandb

def preload_hdf5_to_memory(data_dir: str, file_in: str):
    """
    Load entire HDF5 file into memory as numpy arrays.
    
    Returns:
        Dictionary with keys: 'nodes', 'targets', 'masks', 'n_nodes', 'n_edges'
    """
    print(f"\nPreloading {file_in} into memory...")
    start = time.time()
    
    file_path = os.path.join(data_dir, file_in)
    
    with h5py.File(file_path, 'r') as f:
        sample_ids = list(f.keys())
        n_samples = len(sample_ids)
        
        print(f"Found {n_samples} samples in file")
        
        # Get shapes from first sample
        first_sample = f[sample_ids[0]]
        node_shape = first_sample['padded_nodes'].shape
        target_shape = first_sample['padded_targets'].shape
        mask_shape = first_sample['node_mask'].shape
        edge_shape   = first_sample['padded_edges'].shape
        send_shape   = first_sample['senders'].shape
        recv_shape   = first_sample['receivers'].shape 
        
        print(f"Sample shapes - Nodes: {node_shape}, Edges: {edge_shape}, Targets: {target_shape}, Mask: {mask_shape}")
        
        ### Pre-allocate arrays
        all_nodes = np.zeros((n_samples, *node_shape), dtype=np.float32)
        all_targets = np.zeros((n_samples, *target_shape), dtype=np.float32)
        all_masks = np.zeros((n_samples, *mask_shape), dtype=np.float32)
        all_edges = np.zeros((n_samples, *edge_shape),   dtype=np.float32)
        all_senders = np.zeros((n_samples, *send_shape),   dtype=np.int32)
        all_receivers = np.zeros((n_samples, *recv_shape),   dtype=np.int32)
        all_n_nodes = np.zeros(n_samples, dtype=np.int32)
        all_n_edges = np.zeros(n_samples, dtype=np.int32)
        
        ### Load all samples
        print("Loading samples...", end='', flush=True)
        for i, sample_id in enumerate(sample_ids):
            if i % 10000 == 0 and i > 0:
                print(f"\n  Loaded {i}/{n_samples} samples...", end='', flush=True)
            
            sample = f[sample_id]
            all_nodes[i] = sample['padded_nodes'][:]
            all_targets[i] = sample['padded_targets'][:]
            all_masks[i] = sample['node_mask'][:]
            all_edges[i]     = sample['padded_edges'][:]
            all_senders[i]   = sample['senders'][:]
            all_receivers[i] = sample['receivers'][:]
            all_n_nodes[i] = sample.attrs['n_nodes']
            all_n_edges[i] = sample.attrs['n_edges']
        
        print()
    
    elapsed = time.time() - start
    data_size_gb = (all_nodes.nbytes + all_targets.nbytes + all_masks.nbytes + all_edges.nbytes + all_senders.nbytes + all_receivers.nbytes) / 1e9
    
    print(f"✓ Loaded {n_samples} samples in {elapsed:.2f}s ({data_size_gb:.2f} GB)")
    
    return {
        'nodes': all_nodes,
        'targets': all_targets,
        'masks': all_masks,
        'edges': all_edges,
        'senders': all_senders,
        'receivers': all_receivers,
        'n_nodes': all_n_nodes,
        'n_edges': all_n_edges
    }

def data_loader(
    data_dict: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool = True,
    latent_size: int = 128
) -> Iterator[Tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]]:

    n_samples = len(data_dict['nodes'])
    indices = np.arange(n_samples)

    if shuffle:
        np.random.seed(42) 
        np.random.shuffle(indices)

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]

        batch_graphs = []
        batch_targets = []
        batch_masks = []

        for b, idx in enumerate(batch_indices):
            n_nodes = int(data_dict['n_nodes'][idx])
            n_edges = int(data_dict['n_edges'][idx])

            ### Slice to true sizes
            nodes = jnp.array(data_dict['nodes'][idx])[:n_nodes]
            targets = jnp.array(data_dict['targets'][idx])[:n_nodes]
            masks = jnp.array(data_dict['masks'][idx])[:n_nodes]

            edges = jnp.array(data_dict['edges'][idx])[:n_edges]
            senders = jnp.array(data_dict['senders'][idx])[:n_edges]
            receivers = jnp.array(data_dict['receivers'][idx])[:n_edges]

            graph = jraph.GraphsTuple(
                nodes=nodes,
                edges=edges,
                senders=senders,
                receivers=receivers,
                n_node=jnp.array([n_nodes], dtype=jnp.int32),
                n_edge=jnp.array([n_edges], dtype=jnp.int32),
                globals=jnp.array([[n_nodes / jnp.array(data_dict['masks'][idx]).shape[0]]], dtype=jnp.float32),
            )
            batch_graphs.append(graph)

            batch_targets.append(targets)
            batch_masks.append(masks)

        batched_graph = jraph.batch(batch_graphs)

        batched_targets = jnp.concatenate(batch_targets, axis=0)
        batched_masks = jnp.concatenate(batch_masks, axis=0)

        yield batched_graph, batched_targets, batched_masks

def infinite_data_loader(data_dict, batch_size, latent_size=128, shuffle=True):
    """Yield batches forever; reshuffle each pass."""
    while True:
        yield from data_loader(data_dict, batch_size=batch_size, shuffle=shuffle, latent_size=latent_size)


def create_train_state(model, rng_key, learning_rate, grad_clipping, example_graph):
    
    params = model.init(rng_key, example_graph, deterministic = False)['params']
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clipping), optax.adam(learning_rate))
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

def mse_loss(params, graph, target, mask, apply_fn, training, rng=None):
    deterministic = not training
    kwargs = dict(deterministic=deterministic)
    if not deterministic:
        kwargs["rngs"] = {"dropout": rng}

    preds_graph = apply_fn({'params': params}, graph, **kwargs)
    preds = preds_graph.nodes

    ### Fix shapes depending on output size
    if preds.ndim == 1:
        preds = preds[:, None]
    if target.ndim == 1:
        target = target[:, None]
    if mask.ndim != 1:
        mask = mask.reshape(-1)

    mask_f = mask.astype(preds.dtype)
    count  = mask_f.sum() + 1e-12

    ### Masked MSE
    mse_per_node = ((preds - target) ** 2).sum(-1)
    mse = (mse_per_node * mask_f).sum() / count
    loss = mse

    return loss

def tree_l2_norm(tree):
    return jnp.sqrt(sum([jnp.vdot(x, x) for x in tree_leaves(tree)]))

@jax.jit
def train_step(state, graph, target, mask, rng_key):

    rng, dropout_key, noise_key = jax.random.split(rng_key, 3)

    def loss_fn(params):
        return mse_loss(params, graph, target, mask, state.apply_fn, training=True, rng=dropout_key)

    grads = jax.grad(loss_fn)(state.params)

    return state.apply_gradients(grads=grads)

@jax.jit
def eval_step(state, graph, target, mask):

    loss = mse_loss(state.params, graph, target, mask, state.apply_fn, training=False, rng=None)

    return loss

def train_model(
    train_data: Dict[str, np.ndarray], 
    test_data: Dict[str, np.ndarray], 
    model, 
    batch_size = 128,
    num_train_steps=50_000, 
    eval_every=500, 
    log_every=50, 
    num_eval_batches=None,
    learning_rate=10**-4, 
    grad_clipping = 1, 
    latent_size = 128, 
    early_stopping=False, 
    patience=10, 
    wandb_notes = 'testing',
    hidden_size = None,
    num_mlp_layers = None
    ):

    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)

    example_graph, _, _ = next(data_loader(train_data, batch_size=1, shuffle=False, latent_size=latent_size))
    state = create_train_state(model, init_key, learning_rate, grad_clipping, example_graph)
    
    train_losses = []
    test_losses = []

    ### Early stopping initial quantities
    best_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    ### Tracking with WandB
    run = wandb.init(
        entity="erichabjan-northeastern-university",
        project="phase-space-GNN",
        config={
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "num_mlp_layers": num_mlp_layers,
            "architecture": "ConvGNN",
            "notes": wandb_notes
            },
            )
    
    train_stream = infinite_data_loader(train_data, batch_size, latent_size=latent_size, shuffle=True)

    def eval_loop():
        total = 0.0
        count = 0
        if num_eval_batches is None:
            # Full pass
            for graph, tgt, mask in data_loader(test_data, batch_size, shuffle=True, latent_size=latent_size):
                loss_val = eval_step(state, graph, tgt, mask)
                total += float(loss_val)
                count += 1
        else:
            # First num_eval_batches batches of the deterministic loader
            it = data_loader(test_data, batch_size, shuffle=True, latent_size=latent_size)
            for _ in range(num_eval_batches):
                try:
                    graph, tgt, mask = next(it)
                except StopIteration:
                    break
                loss_val = eval_step(state, graph, tgt, mask)
                total += float(loss_val)
                count += 1
        return total / max(count, 1)
    
    for step in range(1, num_train_steps + 1):
        graph, tgt, mask = next(train_stream)

        rng_key, batch_key = jax.random.split(rng_key)
        state = train_step(state, graph, tgt, mask, batch_key)

        batch_loss = eval_step(state, graph, tgt, mask)
        batch_loss.block_until_ready()
        batch_loss = float(batch_loss)

        train_losses.append(batch_loss)
        run.log({"Training Loss": batch_loss, "Step": step}, step=step)

        if (step % log_every) == 0:
            print(f"Step {step} | Training Loss: {batch_loss:.6f}")
        
        if (step % eval_every) == 0:
            val_loss = eval_loop()
            test_losses.append(val_loss)

            print(f"Step {step} | Validation Loss: {val_loss:.6f}")
            run.log({"Validation Loss": val_loss, "Step": step}, step=step)

            ### Early stopping
            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = state
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1

                if evals_without_improvement >= patience:
                    print(f"Early stopping triggered at step {step}")
                    if best_state is not None:
                        state = best_state
                    break

    run.finish()
    
    return state, model, np.array(train_losses), np.array(test_losses)


def predict(model, params, data_dir, data_prefix = 'test'):
    """
    Make predictions with a trained model.
    """
        
    predictions = []
    mask_arr = []
    tgt_arr = []
        
    ### Predictions batch by batch
    for graph, tgt, mask in create_dataloader(data_dir, data_prefix, shuffle=False):

        preds = model.apply({'params': params}, graph, deterministic = True)

        predictions.append(preds.nodes)
        tgt_arr.append(tgt)
        mask_arr.append(mask)
        
    return jnp.concatenate(predictions, axis=0).squeeze(), jnp.concatenate(tgt_arr, axis=0).squeeze(), jnp.concatenate(mask_arr, axis=0).squeeze()