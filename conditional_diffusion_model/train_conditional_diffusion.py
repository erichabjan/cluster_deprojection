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
    print(f"\nPreloading {file_path} into memory...")
    start = time.time()

    with h5py.File(file_path, "r") as f:
        sample_ids = list(f.keys())
        n_samples = len(sample_ids)
        first = f[sample_ids[0]]

        images_shape = first["images"].shape
        cube_shape = first["density_cube"].shape
        feat_shape = first["gal_features"].shape
        pix_shape = first["gal_pixel_coords"].shape
        mask_shape = first["mask"].shape

        images = np.zeros((n_samples, *images_shape), dtype=np.float32)
        cubes = np.zeros((n_samples, *cube_shape), dtype=np.float32)
        gal_features = np.zeros((n_samples, *feat_shape), dtype=np.float32)
        gal_pixel_coords = np.zeros((n_samples, *pix_shape), dtype=np.float32)
        mask = np.zeros((n_samples, *mask_shape), dtype=np.float32)

        for i, sid in enumerate(sample_ids):
            g = f[sid]
            images[i] = g["images"][:]
            cubes[i] = g["density_cube"][:]
            gal_features[i] = g["gal_features"][:]
            gal_pixel_coords[i] = g["gal_pixel_coords"][:]
            mask[i] = g["mask"][:]

        metadata = {k: f.attrs[k] for k in f.attrs.keys()}

    elapsed = time.time() - start
    size_gb = (
        images.nbytes
        + cubes.nbytes
        + gal_features.nbytes
        + gal_pixel_coords.nbytes
        + mask.nbytes
    ) / 1e9
    print(f"✓ Loaded {n_samples} samples in {elapsed:.2f}s ({size_gb:.2f} GB)")

    return dict(
        images=images,
        cubes=cubes,
        gal_features=gal_features,
        gal_pixel_coords=gal_pixel_coords,
        mask=mask,
        metadata=metadata,
    )


def data_loader(
    data: Dict[str, np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    n = data["images"].shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)

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

        yield (
            jnp.asarray(images),
            jnp.asarray(cubes),
            jnp.asarray(gal_features),
            jnp.asarray(gal_pixel_coords),
            jnp.asarray(mask),
        )


def infinite_data_loader(data, batch_size, rng, shuffle=True):
    while True:
        yield from data_loader(data, batch_size=batch_size, rng=rng, shuffle=shuffle)


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
    images_ex, cubes_ex, gal_features_ex, gal_pixel_coords_ex, mask_ex = example_batch
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
    timesteps,
    noise,
    alpha_bars,
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
    )
    return jnp.mean((pred_noise - noise) ** 2)


@jax.jit
def train_step(state, images, cubes, gal_features, gal_pixel_coords, mask, rng_key, alpha_bars, num_timesteps):
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
            t,
            noise,
            alpha_bars,
        )

    grads = jax.grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    loss = loss_fn(new_state.params)
    return new_state, loss


@jax.jit
def eval_step(state, images, cubes, gal_features, gal_pixel_coords, mask, rng_key, alpha_bars, num_timesteps):
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

    train_stream = infinite_data_loader(train_data, batch_size, rng=train_rng, shuffle=True)
    example_batch = next(
        data_loader(
            train_data,
            batch_size=min(batch_size, 2),
            rng=train_rng,
            shuffle=False,
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
        it = data_loader(test_data, batch_size=batch_size, rng=test_rng, shuffle=True)

        if num_eval_batches is None:
            for images, cubes, gal_features, gal_pixel_coords, mask in it:
                nonlocal_rng = jax.random.PRNGKey(1000 + count)
                loss = eval_step(
                    state, images, cubes, gal_features, gal_pixel_coords, mask,
                    nonlocal_rng, alpha_bars, num_diffusion_steps
                )
                total += float(loss)
                count += 1
        else:
            for k in range(num_eval_batches):
                try:
                    images, cubes, gal_features, gal_pixel_coords, mask = next(it)
                except StopIteration:
                    break
                nonlocal_rng = jax.random.PRNGKey(1000 + k)
                loss = eval_step(
                    state, images, cubes, gal_features, gal_pixel_coords, mask,
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
        images, cubes, gal_features, gal_pixel_coords, mask = next(train_stream)
        rng_key, step_key = jax.random.split(rng_key)

        state, loss_train = train_step(
            state,
            images,
            cubes,
            gal_features,
            gal_pixel_coords,
            mask,
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