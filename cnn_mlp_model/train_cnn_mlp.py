from typing import Dict, Iterator, Tuple, Optional
import os
import time
import h5py
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

import wandb


def preload_hdf5_to_memory(file_path: str) -> Dict[str, np.ndarray]:
    """
    Loads all samples from preprocessed HDF5 into RAM.

    Expected datasets per sample:
      images:           (C,H,W)      e.g. (3,128,128)
      gal_features:     (MAX,4)
      gal_pixel_coords: (MAX,2)
      targets:          (MAX,D)      e.g. D=1 or D=3
      mask:             (MAX,)
    """
    print(f"\nPreloading {file_path} into memory...")
    start = time.time()

    with h5py.File(file_path, "r") as f:
        sample_ids = list(f.keys())
        n_samples = len(sample_ids)
        first = f[sample_ids[0]]

        images_shape = first["images"].shape
        feat_shape = first["gal_features"].shape
        pix_shape = first["gal_pixel_coords"].shape
        targ_shape = first["targets"].shape
        mask_shape = first["mask"].shape

        images = np.zeros((n_samples, *images_shape), dtype=np.float32)
        gal_features = np.zeros((n_samples, *feat_shape), dtype=np.float32)
        gal_pixel_coords = np.zeros((n_samples, *pix_shape), dtype=np.float32)
        targets = np.zeros((n_samples, *targ_shape), dtype=np.float32)
        mask = np.zeros((n_samples, *mask_shape), dtype=np.float32)

        for i, sid in enumerate(sample_ids):
            g = f[sid]
            images[i] = g["images"][:]
            gal_features[i] = g["gal_features"][:]
            gal_pixel_coords[i] = g["gal_pixel_coords"][:]
            targets[i] = g["targets"][:]
            mask[i] = g["mask"][:]

    elapsed = time.time() - start
    size_gb = (
        images.nbytes
        + gal_features.nbytes
        + gal_pixel_coords.nbytes
        + targets.nbytes
        + mask.nbytes
    ) / 1e9
    print(f"✓ Loaded {n_samples} samples in {elapsed:.2f}s ({size_gb:.2f} GB)")

    return dict(
        images=images,
        gal_features=gal_features,
        gal_pixel_coords=gal_pixel_coords,
        targets=targets,
        mask=mask,
    )


def data_loader(
    data: Dict[str, np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Yields batches of *galaxies* (not clusters).

    Each batch item is a galaxy sampled from a sample (cluster-projection):
      images:           (B, H, W, C)
      mlp_in:           (B, 1, Fm)
      gal_pixel_coords: (B, 1, 2)
      targets:          (B, 1, D)

    Notes:
    - HDF5 stores images channels-first (C,H,W), so we transpose to (H,W,C).
    - We add a singleton N dimension because the model is now per-cluster/per-galaxy:
        mlp_in / coords / targets -> shape (B, N=1, ...)
    """
    valid = np.argwhere(data["mask"] > 0.5)
    if shuffle:
        rng.shuffle(valid)

    for i in range(0, len(valid), batch_size):
        chunk = valid[i:i + batch_size]
        if len(chunk) == 0:
            continue

        s_idx = chunk[:, 0]
        n_idx = chunk[:, 1]

        images = data["images"][s_idx]                  # (B, C, H, W)
        images = np.transpose(images, (0, 2, 3, 1))     # (B, H, W, C)

        mlp_in = data["gal_features"][s_idx, n_idx]         # (B, Fm)
        pix = data["gal_pixel_coords"][s_idx, n_idx]        # (B, 2)
        targets = data["targets"][s_idx, n_idx]             # (B, D)

        mlp_in = mlp_in[:, None, :]      # (B,1,Fm)
        pix = pix[:, None, :]            # (B,1,2)
        targets = targets[:, None, :]    # (B,1,D)

        yield (
            jnp.asarray(images),
            jnp.asarray(mlp_in),
            jnp.asarray(pix),
            jnp.asarray(targets),
        )


def infinite_data_loader(data, batch_size, rng, shuffle=True):
    while True:
        yield from data_loader(data, batch_size=batch_size, rng=rng, shuffle=shuffle)


def create_train_state(model, rng_key, learning_rate, grad_clipping, example_batch):
    images_ex, mlp_ex, pix_ex, _ = example_batch
    params = model.init(
        rng_key,
        images_ex,
        mlp_ex,
        pix_ex,
        deterministic=False,
    )["params"]

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clipping),
        optax.adam(learning_rate),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def mse_loss(params, apply_fn, images, mlp_in, gal_pixel_coords, targets, training: bool, rng=None):
    """
    Standard MSE over the batch.

    Shapes:
      preds, targets: (B,1,D) in this trainer (galaxy-batched with singleton N)
    """
    deterministic = not training
    kwargs = dict(deterministic=deterministic)
    if not deterministic:
        kwargs["rngs"] = {"dropout": rng}

    preds = apply_fn({"params": params}, images, mlp_in, gal_pixel_coords, **kwargs)  # (B,1,D)

    return jnp.mean(jnp.sum((preds - targets) ** 2, axis=-1))


@jax.jit
def train_step(state, images, mlp_in, gal_pixel_coords, targets, rng_key):
    rng, dropout_key = jax.random.split(rng_key, 2)

    def loss_fn(params):
        return mse_loss(
            params,
            state.apply_fn,
            images,
            mlp_in,
            gal_pixel_coords,
            targets,
            training=True,
            rng=dropout_key,
        )

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


@jax.jit
def eval_step(state, images, mlp_in, gal_pixel_coords, targets):
    return mse_loss(
        state.params,
        state.apply_fn,
        images,
        mlp_in,
        gal_pixel_coords,
        targets,
        training=False,
        rng=None,
    )


def train_model(
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    model,
    batch_size: int = 256,
    num_train_steps: int = 50_000,
    eval_every: int = 500,
    log_every: int = 50,
    num_eval_batches: Optional[int] = 200,
    learning_rate=1e-3,
    grad_clipping: float = 1.0,
    wandb_project: str = "phase-space-CNN-MLP",
    wandb_notes: str = "",
    cfg_dict: Optional[dict] = None,
):
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)

    train_rng = np.random.default_rng(42)
    test_rng = np.random.default_rng(123)

    train_stream = infinite_data_loader(train_data, batch_size, rng=train_rng, shuffle=True)

    example_batch = next(data_loader(train_data, batch_size=8, rng=train_rng, shuffle=False))
    state = create_train_state(model, init_key, learning_rate, grad_clipping, example_batch)

    run = wandb.init(
        entity="erichabjan-northeastern-university",
        project=wandb_project,
        config=dict(
            learning_rate=str(learning_rate),
            grad_clipping=float(grad_clipping),
            batch_size=int(batch_size),
            notes=wandb_notes,
            **({} if cfg_dict is None else cfg_dict),
        ),
    )

    def eval_loop():
        total = 0.0
        count = 0
        it = data_loader(test_data, batch_size=batch_size, rng=test_rng, shuffle=True)
        if num_eval_batches is None:
            for images, mlp_in, gal_pixel_coords, targets in it:
                loss = eval_step(state, images, mlp_in, gal_pixel_coords, targets)
                total += float(loss)
                count += 1
        else:
            for _ in range(num_eval_batches):
                try:
                    images, mlp_in, gal_pixel_coords, targets = next(it)
                except StopIteration:
                    break
                loss = eval_step(state, images, mlp_in, gal_pixel_coords, targets)
                total += float(loss)
                count += 1
        return total / max(count, 1)

    train_losses = []
    test_losses = []

    for step in range(1, num_train_steps + 1):
        images, mlp_in, gal_pixel_coords, targets = next(train_stream)

        rng_key, step_key = jax.random.split(rng_key)
        state = train_step(state, images, mlp_in, gal_pixel_coords, targets, step_key)

        loss_train = eval_step(state, images, mlp_in, gal_pixel_coords, targets)
        loss_train.block_until_ready()
        loss_train = float(loss_train)
        train_losses.append(loss_train)
        run.log({"train_loss": loss_train, "step": step}, step=step)

        if step % log_every == 0:
            print(f"Step {step} | train_loss: {loss_train:.6f}")

        if step % eval_every == 0:
            loss_val = eval_loop()
            test_losses.append(loss_val)
            print(f"Step {step} | val_loss: {loss_val:.6f}")
            run.log({"val_loss": loss_val, "step": step}, step=step)

    run.finish()
    return state, np.asarray(train_losses), np.asarray(test_losses)