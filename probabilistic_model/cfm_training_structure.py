import os
import time
import h5py
import numpy as np
from typing import Iterator, Tuple, Dict, Optional

import jax
import jax.numpy as jnp
import optax
import jraph
from flax.training import train_state
import wandb
import functools


def preload_hdf5_to_memory(data_dir: str, file_in: str):
    print(f"\nPreloading {file_in} into memory...")
    start = time.time()

    file_path = os.path.join(data_dir, file_in)
    with h5py.File(file_path, "r") as f:
        sample_ids = list(f.keys())
        n_samples = len(sample_ids)
        print(f"Found {n_samples} samples in file")

        first = f[sample_ids[0]]
        node_shape = first["padded_nodes"].shape
        target_shape = first["padded_targets"].shape
        mask_shape = first["node_mask"].shape
        edge_shape = first["padded_edges"].shape
        send_shape = first["senders"].shape
        recv_shape = first["receivers"].shape

        print(f"Sample shapes - Nodes: {node_shape}, Edges: {edge_shape}, Targets: {target_shape}, Mask: {mask_shape}")

        all_nodes = np.zeros((n_samples, *node_shape), dtype=np.float32)
        all_targets = np.zeros((n_samples, *target_shape), dtype=np.float32)
        all_masks = np.zeros((n_samples, *mask_shape), dtype=np.float32)
        all_edges = np.zeros((n_samples, *edge_shape), dtype=np.float32)
        all_senders = np.zeros((n_samples, *send_shape), dtype=np.int32)
        all_receivers = np.zeros((n_samples, *recv_shape), dtype=np.int32)
        all_n_nodes = np.zeros(n_samples, dtype=np.int32)
        all_n_edges = np.zeros(n_samples, dtype=np.int32)

        print("Loading samples...", end="", flush=True)
        for i, sid in enumerate(sample_ids):
            if i % 10000 == 0 and i > 0:
                print(f"\n  Loaded {i}/{n_samples} samples...", end="", flush=True)

            s = f[sid]
            all_nodes[i] = s["padded_nodes"][:]
            all_targets[i] = s["padded_targets"][:]
            all_masks[i] = s["node_mask"][:]
            all_edges[i] = s["padded_edges"][:]
            all_senders[i] = s["senders"][:]
            all_receivers[i] = s["receivers"][:]
            all_n_nodes[i] = s.attrs["n_nodes"]
            all_n_edges[i] = s.attrs["n_edges"]
        print()

    elapsed = time.time() - start
    data_size_gb = (all_nodes.nbytes + all_targets.nbytes + all_masks.nbytes +
                    all_edges.nbytes + all_senders.nbytes + all_receivers.nbytes) / 1e9
    print(f"✓ Loaded {n_samples} samples in {elapsed:.2f}s ({data_size_gb:.2f} GB)")

    return {
        "nodes": all_nodes,
        "targets": all_targets,
        "masks": all_masks,
        "edges": all_edges,
        "senders": all_senders,
        "receivers": all_receivers,
        "n_nodes": all_n_nodes,
        "n_edges": all_n_edges,
    }


def data_loader(
    data_dict: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[Tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]]:
    n_samples = len(data_dict["nodes"])
    indices = np.arange(n_samples)
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]

        batch_graphs = []
        batch_targets = []
        batch_masks = []

        for idx in batch_indices:
            n_nodes = int(data_dict["n_nodes"][idx])
            n_edges = int(data_dict["n_edges"][idx])

            nodes = jnp.array(data_dict["nodes"][idx])[:n_nodes]
            targets = jnp.array(data_dict["targets"][idx])[:n_nodes]
            masks = jnp.array(data_dict["masks"][idx])[:n_nodes]

            edges = jnp.array(data_dict["edges"][idx])[:n_edges]
            senders = jnp.array(data_dict["senders"][idx])[:n_edges]
            receivers = jnp.array(data_dict["receivers"][idx])[:n_edges]

            graph = jraph.GraphsTuple(
                nodes=nodes,
                edges=edges,
                senders=senders,
                receivers=receivers,
                n_node=jnp.array([n_nodes], dtype=jnp.int32),
                n_edge=jnp.array([n_edges], dtype=jnp.int32),
                globals=jnp.array([[n_nodes / jnp.array(data_dict["masks"][idx]).shape[0]]], dtype=jnp.float32),
            )
            batch_graphs.append(graph)
            batch_targets.append(targets)
            batch_masks.append(masks)

        batched_graph = jraph.batch(batch_graphs)
        batched_targets = jnp.concatenate(batch_targets, axis=0)
        batched_masks = jnp.concatenate(batch_masks, axis=0)
        yield batched_graph, batched_targets, batched_masks


def infinite_data_loader(data_dict, batch_size, shuffle=True):
    while True:
        yield from data_loader(data_dict, batch_size=batch_size, shuffle=shuffle)


def create_train_state(model, rng_key, learning_rate, grad_clipping, example_graph, target_dim: int = 3):

    n_total = example_graph.nodes.shape[0]
    x_t0 = jnp.zeros((n_total, target_dim), dtype=jnp.float32)
    t0 = jnp.zeros((n_total, 1), dtype=jnp.float32)

    params = model.init(rng_key, example_graph, x_t0, t0, deterministic=False)["params"]
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clipping),
        optax.adam(learning_rate),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


def cfm_loss(params, model_apply_fn, graph, x1, mask, rng_key, training: bool, target_dim: int = 3):

    deterministic = not training
    rng_key, t_key, x0_key, dropout_key = jax.random.split(rng_key, 4)

    ### Sample time
    t = jax.random.uniform(t_key, shape=(x1.shape[0], 1), minval=0.0, maxval=1.0)

    ### Noise
    x0 = jax.random.normal(x0_key, shape=x1.shape, dtype=x1.dtype)

    ### Optimal transport conditional flow
    x_t = (1.0 - t) * x0 + t * x1

    ### Target velocity field
    u_t = x1 - x0

    kwargs = dict(deterministic=deterministic)
    if not deterministic:
        kwargs["rngs"] = {"dropout": dropout_key}

    v_pred = model_apply_fn({"params": params}, graph, x_t, t, **kwargs)

    ### CFM objective
    mask_f = mask.astype(v_pred.dtype).reshape(-1)
    count = mask_f.sum() + 1e-12
    mse_per_node = jnp.sum((v_pred - u_t) ** 2, axis=-1)
    loss = jnp.sum(mse_per_node * mask_f) / count
    return loss


@jax.jit
def train_step(state, graph, targets, mask, rng_key, target_dim: int = 3):
    def loss_fn(params):
        return cfm_loss(params, state.apply_fn, graph, targets, mask, rng_key, training=True, target_dim=target_dim)

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


@jax.jit
def eval_step(state, graph, targets, mask, rng_key, target_dim: int = 3):
    return cfm_loss(state.params, state.apply_fn, graph, targets, mask, rng_key, training=False, target_dim=target_dim)


@functools.partial(jax.jit, static_argnames=("apply_fn",))
def euler_step(params, apply_fn, graph, x, t, dt):

    t_vec = jnp.full((x.shape[0], 1), t, dtype=x.dtype)
    v = apply_fn({"params": params}, graph, x, t_vec, deterministic=True)

    return x + dt * v


def sample_cfm(
    params,
    apply_fn,
    graph,
    rng_key,
    num_steps: int = 64,
    target_dim: int = 3,
):

    n_total = graph.nodes.shape[0]
    x = jax.random.normal(rng_key, (n_total, target_dim), dtype=jnp.float32)

    ts = jnp.linspace(0.0, 1.0, num_steps + 1)
    dt = ts[1] - ts[0]

    for k in range(num_steps):
        x = euler_step(params, apply_fn, graph, x, ts[k], dt)

    return x


def train_model(
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    model,
    batch_size: int = 4,
    num_train_steps: int = 50_000,
    eval_every: int = 25,
    log_every: int = 50,
    num_eval_batches: Optional[int] = 100,
    learning_rate: float = 3e-5,
    grad_clipping: float = 1.0,
    early_stopping: bool = True,
    patience: int = 200,
    wandb_notes: str = "cfm run",
    hidden_size: Optional[int] = None,
    num_mlp_layers: Optional[int] = None,
    target_dim: int = 3,
):
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)

    example_graph, _, _ = next(data_loader(train_data, batch_size=1, shuffle=False))
    state = create_train_state(model, init_key, learning_rate, grad_clipping, example_graph, target_dim=target_dim)

    train_losses = []
    test_losses = []

    best_loss = float("inf")
    best_state = None
    evals_wo_improve = 0

    run = wandb.init(
        entity="erichabjan-northeastern-university",
        project="phase-space-GNN",
        config={
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "num_mlp_layers": num_mlp_layers,
            "architecture": "CFM-ConvGNN",
            "notes": wandb_notes,
            "target_dim": target_dim,
        },
    )

    train_stream = infinite_data_loader(train_data, batch_size=batch_size, shuffle=True)

    def eval_loop(curr_state, rng_in):
        total = 0.0
        count = 0
        it = data_loader(test_data, batch_size=batch_size, shuffle=True)
        for b in range(num_eval_batches if num_eval_batches is not None else 10**12):
            try:
                graph, tgt, mask = next(it)
            except StopIteration:
                break
            rng_in, k = jax.random.split(rng_in)
            loss_val = eval_step(curr_state, graph, tgt, mask, k, target_dim=target_dim)
            total += float(loss_val)
            count += 1
        return total / max(count, 1)

    for step in range(1, num_train_steps + 1):
        graph, tgt, mask = next(train_stream)

        rng_key, step_key = jax.random.split(rng_key)
        state = train_step(state, graph, tgt, mask, step_key, target_dim=target_dim)

        rng_key, log_key = jax.random.split(rng_key)
        batch_loss = eval_step(state, graph, tgt, mask, log_key, target_dim=target_dim)
        batch_loss.block_until_ready()
        batch_loss = float(batch_loss)

        train_losses.append(batch_loss)
        run.log({"Training Loss": batch_loss, "Step": step}, step=step)

        if (step % log_every) == 0:
            print(f"Step {step} | Training Loss: {batch_loss:.6f}")

        if (step % eval_every) == 0:
            rng_key, ev_key = jax.random.split(rng_key)
            val_loss = eval_loop(state, ev_key)
            test_losses.append(val_loss)

            print(f"Step {step} | Validation Loss: {val_loss:.6f}")
            run.log({"Validation Loss": val_loss, "Step": step}, step=step)

            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = state
                    evals_wo_improve = 0
                else:
                    evals_wo_improve += 1

                if evals_wo_improve >= patience:
                    print(f"Early stopping triggered at step {step}")
                    if best_state is not None:
                        state = best_state
                    break

    run.finish()
    return state, model, np.array(train_losses), np.array(test_losses)
