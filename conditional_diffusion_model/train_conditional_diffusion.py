from typing import Dict, Iterator, Tuple, Optional
import time
import h5py
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

import os
import pickle
from flax import serialization

import wandb


def preload_hdf5_to_memory(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load a full dataset file into memory.

    Fixed-shape data (images, cubes, member point cloud, targets) are stacked
    into dense (n_samples, ...) arrays. The weak-lensing source point clouds
    from the shape_dynamics files are RAGGED (variable N per sample) and are
    returned as lists of arrays:
      src_features [i]: (N_i, 7) float32  e1, e2, ix, iy, z_s, z_lens, cell_frac
      src_cell_id  [i]: (N_i,)   int32    iy*S + ix, rows pre-sorted by this
      src_cell_ptr     : (n_samples, S*S+1) int32 CSR pointers; sample i's
                         cell c holds src_features[i][ptr[i,c]:ptr[i,c+1]]
    with S = metadata["shape_grid_size"]. Old h5 files without source data
    load fine; the src_* / z_lens / n_sources keys are simply absent.
    """
    print(f"\nPreloading {file_path} into memory...")
    start = time.time()

    with h5py.File(file_path, "r") as f:
        sample_ids = sorted(f.keys())
        n_samples = len(sample_ids)
        first = f[sample_ids[0]]

        images = np.zeros((n_samples, *first["images"].shape), dtype=np.float32)
        cubes = np.zeros((n_samples, *first["density_cube"].shape), dtype=np.float32)
        gal_features = np.zeros((n_samples, *first["gal_features"].shape), dtype=np.float32)
        gal_targets = np.zeros((n_samples, *first["gal_targets"].shape), dtype=np.float32)
        gal_pixel_coords = np.zeros((n_samples, *first["gal_pixel_coords"].shape), dtype=np.float32)
        mask = np.zeros((n_samples, *first["mask"].shape), dtype=np.float32)
        globals_target = np.zeros((n_samples, *first["globals_target"].shape), dtype=np.float32)

        cluster_index = np.zeros((n_samples,), dtype=np.int32)
        simulation = []

        has_src = "src_features" in first
        if has_src:
            src_features = []
            src_cell_id = []
            src_cell_ptr = np.zeros((n_samples, *first["src_cell_ptr"].shape), dtype=np.int32)
            z_lens = np.zeros((n_samples,), dtype=np.float32)
            n_sources = np.zeros((n_samples,), dtype=np.int64)

        for i, sid in enumerate(sample_ids):
            g = f[sid]
            images[i] = g["images"][:]
            cubes[i] = g["density_cube"][:]
            gal_features[i] = g["gal_features"][:]
            gal_targets[i] = g["gal_targets"][:]
            gal_pixel_coords[i] = g["gal_pixel_coords"][:]
            mask[i] = g["mask"][:]
            globals_target[i] = g["globals_target"][:]
            cluster_index[i] = g.attrs["cluster_index"]
            simulation.append(str(g.attrs["simulation"]))
            if has_src:
                src_features.append(g["src_features"][:])
                src_cell_id.append(g["src_cell_id"][:])
                src_cell_ptr[i] = g["src_cell_ptr"][:]
                z_lens[i] = g.attrs["z_lens"]
                n_sources[i] = g.attrs["n_sources"]

        metadata = {k: f.attrs[k] for k in f.attrs.keys()}

    elapsed = time.time() - start
    size_b = (
        images.nbytes
        + cubes.nbytes
        + gal_features.nbytes
        + gal_targets.nbytes
        + gal_pixel_coords.nbytes
        + mask.nbytes
        + globals_target.nbytes
    )
    if has_src:
        size_b += (sum(a.nbytes for a in src_features)
                   + sum(a.nbytes for a in src_cell_id)
                   + src_cell_ptr.nbytes)
    print(f"✓ Loaded {n_samples} samples in {elapsed:.2f}s ({size_b / 1e9:.2f} GB)")
    if has_src:
        print(f"  sources/sample: min={n_sources.min()}, "
              f"median={int(np.median(n_sources))}, max={n_sources.max()}; "
              f"shape grid {int(metadata['shape_grid_size'])}"
              f"x{int(metadata['shape_grid_size'])}")

    out = dict(
        images=images,
        cubes=cubes,
        gal_features=gal_features,
        gal_targets=gal_targets,
        gal_pixel_coords=gal_pixel_coords,
        mask=mask,
        globals_target=globals_target,
        cluster_index=cluster_index,
        simulation=np.array(simulation),
        metadata=metadata,
    )
    if has_src:
        out.update(
            src_features=src_features,
            src_cell_id=src_cell_id,
            src_cell_ptr=src_cell_ptr,
            z_lens=z_lens,
            n_sources=n_sources,
        )
    return out


def make_src_batch(
    data: Dict[str, np.ndarray],
    bidx: np.ndarray,
    n_src_pad: int,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble the ragged source point clouds of one batch into fixed-shape
    arrays for jit:
      src_features (B, n_src_pad, 7) zero-padded
      src_cell_id  (B, n_src_pad)    padded rows get the dummy cell S*S,
                                     which SourceShapeEncoder drops
      src_log_count(B, S, S, 1)      log1p(sources per cell), from cell_ptr
    """
    B = len(bidx)
    S = grid_size
    feats = np.zeros((B, n_src_pad, 7), dtype=np.float32)
    cids = np.full((B, n_src_pad), S * S, dtype=np.int32)
    logc = np.zeros((B, S, S, 1), dtype=np.float32)
    for j, i in enumerate(bidx):
        f = data["src_features"][i]
        n = f.shape[0]
        if n > n_src_pad:
            raise ValueError(
                f"sample {i} has {n} sources > n_src_pad={n_src_pad}; "
                "recompute n_src_pad as the max over all splits"
            )
        feats[j, :n] = f
        cids[j, :n] = data["src_cell_id"][i]
        counts = np.diff(data["src_cell_ptr"][i]).reshape(S, S)
        logc[j, ..., 0] = np.log1p(counts)
    return feats, cids, logc


def data_loader(
    data: Dict[str, np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
    n_src_pad: Optional[int] = None,
    grid_size: Optional[int] = None,
) -> Iterator[Tuple[jnp.ndarray, ...]]:
    n = data["images"].shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)

    # Padded source-cloud size / shape grid; pass shared values across splits
    # so train and eval batches have identical shapes (single jit compile).
    if n_src_pad is None:
        n_src_pad = int(np.max(data["n_sources"]))
    if grid_size is None:
        grid_size = int(data["metadata"]["shape_grid_size"])

    for i in range(0, n, batch_size):
        bidx = idx[i:i + batch_size]
        if len(bidx) == 0:
            continue

        images = data["images"][bidx]                    # (B,C,H,W)
        images = np.transpose(images, (0, 2, 3, 1))     # (B,H,W,C)

        cubes = data["cubes"][bidx]                      # (B,Z,Y,X)
        cubes = cubes[..., None]                         # (B,Z,Y,X,1)

        gal_features = data["gal_features"][bidx]        # (B,N,4)
        gal_pixel_coords = data["gal_pixel_coords"][bidx]# (B,N,2)
        mask = data["mask"][bidx]                        # (B,N)

        src_features, src_cell_id, src_log_count = make_src_batch(
            data, bidx, n_src_pad, grid_size)

        yield (
            jnp.asarray(images),
            jnp.asarray(cubes),
            jnp.asarray(gal_features),
            jnp.asarray(gal_pixel_coords),
            jnp.asarray(mask),
            jnp.asarray(src_features),
            jnp.asarray(src_cell_id),
            jnp.asarray(src_log_count),
        )


def infinite_data_loader(data, batch_size, rng, shuffle=True,
                         n_src_pad=None, grid_size=None):
    while True:
        yield from data_loader(data, batch_size=batch_size, rng=rng, shuffle=shuffle,
                               n_src_pad=n_src_pad, grid_size=grid_size)


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    betas = jnp.linspace(beta_start, beta_end, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0)
    return betas, alphas, alpha_bars


def q_sample(x0, t, noise, alpha_bars):
    a_bar = alpha_bars[t]
    while a_bar.ndim < x0.ndim:
        a_bar = a_bar[..., None]
    return jnp.sqrt(a_bar) * x0 + jnp.sqrt(1.0 - a_bar) * noise


def create_train_state(model, rng_key, learning_rate, grad_clipping, example_batch):
    (images_ex, cubes_ex, gal_features_ex, gal_pixel_coords_ex, mask_ex,
     src_features_ex, src_cell_id_ex, src_log_count_ex) = example_batch
    B = cubes_ex.shape[0]
    t_ex = jnp.zeros((B,), dtype=jnp.int32)

    params = model.init(
        rng_key,
        noisy_cube=cubes_ex,
        timesteps=t_ex,
        cond_images=images_ex,
        gal_features=gal_features_ex,
        gal_pixel_coords=gal_pixel_coords_ex,
        gal_mask=mask_ex,
        src_features=src_features_ex,
        src_cell_id=src_cell_id_ex,
        src_log_count=src_log_count_ex,
    )["params"]

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clipping),
        optax.adam(learning_rate),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def diffusion_loss(
    params,
    apply_fn,
    images,
    cubes,
    gal_features,
    gal_pixel_coords,
    mask,
    src_features,
    src_cell_id,
    src_log_count,
    timesteps,
    noise,
    alpha_bars,
    floor_value=-5.0,
    bg_weight=0.02,
    fg_weight=1.0,
):
    noisy = q_sample(cubes, timesteps, noise, alpha_bars)
    pred_noise = apply_fn(
        {"params": params},
        noisy_cube=noisy,
        timesteps=timesteps,
        cond_images=images,
        gal_features=gal_features,
        gal_pixel_coords=gal_pixel_coords,
        gal_mask=mask,
        src_features=src_features,
        src_cell_id=src_cell_id,
        src_log_count=src_log_count,
    )
    #weights = jnp.where(cubes <= floor_value + 1e-6, bg_weight, fg_weight)
    #sqerr = (pred_noise - noise) ** 2
    return jnp.mean((pred_noise - noise) ** 2)
    #return jnp.sum(weights * sqerr) / jnp.sum(weights)


@jax.jit
def train_step(state, images, cubes, gal_features, gal_pixel_coords, mask,
               src_features, src_cell_id, src_log_count,
               rng_key, alpha_bars, num_timesteps):
    rng_t, rng_n = jax.random.split(rng_key, 2)
    B = cubes.shape[0]
    t = jax.random.randint(rng_t, shape=(B,), minval=0, maxval=num_timesteps)
    noise = jax.random.normal(rng_n, shape=cubes.shape)

    def loss_fn(params):
        return diffusion_loss(
            params,
            state.apply_fn,
            images,
            cubes,
            gal_features,
            gal_pixel_coords,
            mask,
            src_features,
            src_cell_id,
            src_log_count,
            t,
            noise,
            alpha_bars,
        )

    grads = jax.grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    loss = loss_fn(new_state.params)
    return new_state, loss


@jax.jit
def eval_step(state, images, cubes, gal_features, gal_pixel_coords, mask,
              src_features, src_cell_id, src_log_count,
              rng_key, alpha_bars, num_timesteps):
    rng_t, rng_n = jax.random.split(rng_key, 2)
    B = cubes.shape[0]
    t = jax.random.randint(rng_t, shape=(B,), minval=0, maxval=num_timesteps)
    noise = jax.random.normal(rng_n, shape=cubes.shape)

    return diffusion_loss(
        state.params,
        state.apply_fn,
        images,
        cubes,
        gal_features,
        gal_pixel_coords,
        mask,
        src_features,
        src_cell_id,
        src_log_count,
        t,
        noise,
        alpha_bars,
    )


def sample_ddpm(
    model_apply,
    params,
    cond_images,
    gal_features,
    gal_pixel_coords,
    mask,
    src_features,
    src_cell_id,
    src_log_count,
    sample_shape,
    rng_key,
    betas,
    alphas,
    alpha_bars,
):
    """
    cond_images:      (B,H,W,C)
    gal_features:     (B,N,4)
    gal_pixel_coords: (B,N,2)
    mask:             (B,N)
    src_features:     (B,Ns,7)
    src_cell_id:      (B,Ns)
    src_log_count:    (B,S,S,1)
    sample_shape:     (B,Z,Y,X,1)
    """
    x = jax.random.normal(rng_key, shape=sample_shape)

    T = betas.shape[0]
    for step in reversed(range(T)):
        t = jnp.full((sample_shape[0],), step, dtype=jnp.int32)

        pred_noise = model_apply(
            {"params": params},
            noisy_cube=x,
            timesteps=t,
            cond_images=cond_images,
            gal_features=gal_features,
            gal_pixel_coords=gal_pixel_coords,
            gal_mask=mask,
            src_features=src_features,
            src_cell_id=src_cell_id,
            src_log_count=src_log_count,
        )

        beta_t = betas[step]
        alpha_t = alphas[step]
        alpha_bar_t = alpha_bars[step]

        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)

        mean = coef1 * (x - coef2 * pred_noise)

        if step > 0:
            rng_key, subkey = jax.random.split(rng_key)
            z = jax.random.normal(subkey, shape=x.shape)
            sigma = jnp.sqrt(beta_t)
            x = mean + sigma * z
        else:
            x = mean

    return x


def train_model(
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    model,
    batch_size: int = 2,
    num_train_steps: int = 100_000,
    eval_every: int = 250,
    log_every: int = 50,
    num_eval_batches: Optional[int] = 50,
    learning_rate=1e-4,
    grad_clipping: float = 1.0,
    num_diffusion_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    wandb_project: str = "conditional-3d-diffusion",
    wandb_notes: str = "",
    cfg_dict: Optional[dict] = None,
    checkpoint_dir: str = "./runtime_checkpoints",
    checkpoint_prefix: str = "cond_diffusion_runtime",
    max_runtime_hours: float = 7.8,
    runtime_buffer_minutes: float = 10.0,
):
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)

    train_rng = np.random.default_rng(42)
    test_rng = np.random.default_rng(123)

    os.makedirs(checkpoint_dir, exist_ok=True)

    betas, alphas, alpha_bars = make_beta_schedule(
        num_diffusion_steps, beta_start=beta_start, beta_end=beta_end
    )

    diffusion_cfg = dict(
        betas=np.array(betas),
        alphas=np.array(alphas),
        alpha_bars=np.array(alpha_bars),
        num_diffusion_steps=num_diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    )

    # Shared padded source-cloud size across train and eval so every batch has
    # identical shapes (one jit compile of each step function).
    n_src_pad = int(max(np.max(train_data["n_sources"]), np.max(test_data["n_sources"])))
    grid_size = int(train_data["metadata"]["shape_grid_size"])
    print(f"source clouds padded to n_src_pad={n_src_pad}, shape grid {grid_size}x{grid_size}")

    train_stream = infinite_data_loader(train_data, batch_size, rng=train_rng, shuffle=True,
                                        n_src_pad=n_src_pad, grid_size=grid_size)
    example_batch = next(
        data_loader(
            train_data,
            batch_size=min(batch_size, 2),
            rng=train_rng,
            shuffle=False,
            n_src_pad=n_src_pad,
            grid_size=grid_size,
        )
    )

    state = create_train_state(model, init_key, learning_rate, grad_clipping, example_batch)

    run = wandb.init(
        entity="erichabjan-northeastern-university",
        project=wandb_project,
        config=dict(
            learning_rate=str(learning_rate),
            grad_clipping=float(grad_clipping),
            batch_size=int(batch_size),
            num_diffusion_steps=int(num_diffusion_steps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            notes=wandb_notes,
            **({} if cfg_dict is None else cfg_dict),
        ),
    )

    def save_runtime_checkpoint(state, step, train_losses, test_losses, reason):
        """
        Saves:
          1. full TrainState as flax bytes
          2. metadata / losses / diffusion schedule as pickle
        """
        state_path = os.path.join(
            checkpoint_dir,
            f"{checkpoint_prefix}_state_step{step}.msgpack"
        )
        meta_path = os.path.join(
            checkpoint_dir,
            f"{checkpoint_prefix}_meta_step{step}.pkl"
        )

        with open(state_path, "wb") as f:
            f.write(serialization.to_bytes(state))

        meta = dict(
            step=int(step),
            train_losses=np.asarray(train_losses, dtype=np.float32),
            test_losses=np.asarray(test_losses, dtype=np.float32),
            diffusion_cfg=diffusion_cfg,
            reason=str(reason),
            wandb_run_id=None if run is None else run.id,
        )

        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        print(f"\nSaved runtime checkpoint:")
        print(f"  state: {state_path}")
        print(f"  meta : {meta_path}")
        print(f"  reason: {reason}")

    def eval_loop():
        total = 0.0
        count = 0
        it = data_loader(test_data, batch_size=batch_size, rng=test_rng, shuffle=True,
                         n_src_pad=n_src_pad, grid_size=grid_size)

        if num_eval_batches is None:
            for batch in it:
                nonlocal_rng = jax.random.PRNGKey(1000 + count)
                loss = eval_step(
                    state, *batch,
                    nonlocal_rng, alpha_bars, num_diffusion_steps
                )
                total += float(loss)
                count += 1
        else:
            for k in range(num_eval_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                nonlocal_rng = jax.random.PRNGKey(1000 + k)
                loss = eval_step(
                    state, *batch,
                    nonlocal_rng, alpha_bars, num_diffusion_steps
                )
                total += float(loss)
                count += 1

        return total / max(count, 1)

    train_losses = []
    test_losses = []

    start_time = time.time()
    soft_limit_seconds = max_runtime_hours * 3600.0
    buffer_seconds = runtime_buffer_minutes * 60.0

    for step in range(1, num_train_steps + 1):
        batch = next(train_stream)
        rng_key, step_key = jax.random.split(rng_key)

        state, loss_train = train_step(
            state,
            *batch,
            step_key,
            alpha_bars,
            num_diffusion_steps,
        )
        loss_train.block_until_ready()
        loss_train = float(loss_train)

        train_losses.append(loss_train)
        run.log({"train_loss": loss_train, "step": step}, step=step)

        if step % log_every == 0:
            elapsed_hours = (time.time() - start_time) / 3600.0
            print(f"Step {step} | train_loss: {loss_train:.6f} | elapsed: {elapsed_hours:.2f} hr")

        if step % eval_every == 0:
            loss_val = eval_loop()
            test_losses.append(loss_val)
            print(f"Step {step} | val_loss: {loss_val:.6f}")
            run.log({"val_loss": loss_val, "step": step}, step=step)

        # Check walltime after the step has fully completed
        elapsed_seconds = time.time() - start_time
        time_remaining = soft_limit_seconds - elapsed_seconds

        if time_remaining <= buffer_seconds:
            print(
                f"\nApproaching runtime limit: "
                f"elapsed={elapsed_seconds/3600.0:.2f} hr, "
                f"remaining={time_remaining/60.0:.2f} min"
            )

            # Optional: run one last validation before saving if you want
            # and if this step was not already an eval step.
            if step % eval_every != 0:
                loss_val = eval_loop()
                test_losses.append(loss_val)
                print(f"Final pre-exit val_loss at step {step}: {loss_val:.6f}")
                run.log({"val_loss": loss_val, "step": step}, step=step)

            save_runtime_checkpoint(
                state=state,
                step=step,
                train_losses=train_losses,
                test_losses=test_losses,
                reason="approaching_walltime",
            )

            run.finish()
            return state, np.asarray(train_losses), np.asarray(test_losses), diffusion_cfg

    # Normal completion
    save_runtime_checkpoint(
        state=state,
        step=num_train_steps,
        train_losses=train_losses,
        test_losses=test_losses,
        reason="finished_training",
    )

    run.finish()
    return state, np.asarray(train_losses), np.asarray(test_losses), diffusion_cfg