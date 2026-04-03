import os
import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(os.getcwd())
sys.path.append('/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/cond_diff_old_models')
sys.path.append('/home/habjan.e/TNG/cluster_deprojection/conditional_diffusion_model')


from conditional_diffusion_3d_model import ConditionalUNet3D, DiffusionModelConfig
from train_conditional_diffusion import preload_hdf5_to_memory

suffix = '_64cube_v4'

val_path = "/projects/mccleary_group/habjan.e/TNG/Data/conditional_diffusion_data/cond_diffusion_vdisp_test.h5"

model_dir = "/home/habjan.e/TNG/cluster_deprojection/conditional_diffusion_model/conditional_diffusion_models"
param_path = os.path.join(model_dir, f"cond_diffusion_params{suffix}.pkl")
sched_path = os.path.join(model_dir, f"cond_diffusion_schedule{suffix}.pkl")

out_dir = "/projects/mccleary_group/habjan.e/TNG/Data/conditional_diffusion_data/cd_examples_times"
os.makedirs(out_dir, exist_ok=True)


### Example indices, seeds, inference steps
test_indices = [0, 1, 2, 3]
seeds = np.arange(0, 1, dtype=np.int32)
num_inference_steps_list = [10000, 5000, 1000, 500, 100]


### Load data and model
print("\nLoading validation data...")
data_dict = preload_hdf5_to_memory(val_path)
images_all = data_dict["images"]   # (N, C, H, W)
cubes_all = data_dict["cubes"]     # (N, Z, Y, X)
gal_features_all = data_dict["gal_features"]      # (N, max_nodes, 4)
gal_pixel_coords_all = data_dict["gal_pixel_coords"]  # (N, max_nodes, 2)
mask_all = data_dict["mask"]

print("\nLoading model params and diffusion schedule...")
with open(param_path, "rb") as f:
    params = pickle.load(f)

with open(sched_path, "rb") as f:
    diffusion_cfg = pickle.load(f)

betas_train = jnp.asarray(diffusion_cfg["betas"], dtype=jnp.float32)
alphas_train = jnp.asarray(diffusion_cfg["alphas"], dtype=jnp.float32)
alpha_bars_train = jnp.asarray(diffusion_cfg["alpha_bars"], dtype=jnp.float32)

T_train = len(betas_train)
print(f"Training schedule length T_train = {T_train}")

cfg = DiffusionModelConfig(
    base_channels=32,
    channel_mults=(1, 2, 4),
    time_emb_dim=128,
    out_channels=1,
    galaxy_token_dim=128,
    num_attention_heads=4,
    bottleneck_query_shape=(16, 16, 16),
)

model = ConditionalUNet3D(cfg=cfg)


### Build inference schedule
def make_respaced_inference_schedule(alpha_bars_train, num_inference_steps):
    """
    Construct a new inference schedule with num_inference_steps steps,
    possibly larger or smaller than the training schedule.

    We interpolate in log(alpha_bar) space, then derive alphas/betas.
    We also map each inference step back to a training timestep index
    so the model only sees timestep labels it was trained on.

    Returns
    -------
    betas_inf : jnp.ndarray, shape (T_inf,)
    alphas_inf : jnp.ndarray, shape (T_inf,)
    alpha_bars_inf : jnp.ndarray, shape (T_inf,)
    model_t_indices : jnp.ndarray, shape (T_inf,)
        Integer timestep labels in [0, T_train-1] used as model input.
    """
    alpha_bars_train_np = np.asarray(alpha_bars_train, dtype=np.float64)
    T_train = len(alpha_bars_train_np)
    T_inf = int(num_inference_steps)

    if T_inf < 2:
        raise ValueError("num_inference_steps must be at least 2")

    # Normalized time grids in [0, 1]
    s_train = np.linspace(0.0, 1.0, T_train)
    s_inf = np.linspace(0.0, 1.0, T_inf)

    # Interpolate in log space for stability / smoothness
    log_ab_train = np.log(np.clip(alpha_bars_train_np, 1e-12, 1.0))
    log_ab_inf = np.interp(s_inf, s_train, log_ab_train)
    alpha_bars_inf = np.exp(log_ab_inf)

    # Enforce monotonic decrease just in case
    alpha_bars_inf = np.minimum.accumulate(alpha_bars_inf)

    # Derive alphas and betas from alpha_bars
    alphas_inf = np.empty_like(alpha_bars_inf)
    alphas_inf[0] = alpha_bars_inf[0]
    alphas_inf[1:] = alpha_bars_inf[1:] / alpha_bars_inf[:-1]
    alphas_inf = np.clip(alphas_inf, 1e-8, 1.0)

    betas_inf = 1.0 - alphas_inf
    betas_inf = np.clip(betas_inf, 1e-8, 0.999)

    # Map inference steps to original training timestep labels
    model_t_indices = np.rint(s_inf * (T_train - 1)).astype(np.int32)
    model_t_indices = np.clip(model_t_indices, 0, T_train - 1)

    return (
        jnp.asarray(betas_inf, dtype=jnp.float32),
        jnp.asarray(alphas_inf, dtype=jnp.float32),
        jnp.asarray(alpha_bars_inf, dtype=jnp.float32),
        jnp.asarray(model_t_indices, dtype=jnp.int32),
    )


### Sampler
def sample_ddpm_respaced(
    model_apply,
    params,
    cond_images,
    gal_features,
    gal_pixel_coords,
    gal_mask,
    sample_shape,
    rng_key,
    betas_inf,
    alphas_inf,
    alpha_bars_inf,
    model_t_indices,
):
    """
    cond_images:      (B, H, W, C)
    gal_features:     (B, N, 4)
    gal_pixel_coords: (B, N, 2)
    gal_mask:         (B, N)
    sample_shape:     (B, Z, Y, X, 1)

    betas_inf / alphas_inf / alpha_bars_inf define the inference schedule.
    model_t_indices tells the model which original trained timestep to use.
    """
    x = jax.random.normal(rng_key, shape=sample_shape)

    T_inf = betas_inf.shape[0]

    for step in reversed(range(T_inf)):
        t_model = jnp.full(
            (sample_shape[0],),
            model_t_indices[step],
            dtype=jnp.int32
        )

        pred_noise = model_apply(
            {"params": params},
            noisy_cube=x,
            timesteps=t_model,
            cond_images=cond_images,
            gal_features=gal_features,
            gal_pixel_coords=gal_pixel_coords,
            gal_mask=gal_mask,
        )

        beta_t = betas_inf[step]
        alpha_t = alphas_inf[step]
        alpha_bar_t = alpha_bars_inf[step]

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


def generate_cube_samples_for_schedule(
    model,
    params,
    cond_image_hwc,
    gal_features_n4,
    gal_pixel_coords_n2,
    gal_mask_n,
    seeds,
    cube_shape_zyx,
    alpha_bars_train,
    num_inference_steps,
):
    """
    cond_image_hwc:      (H, W, C)
    gal_features_n4:     (Ngal_max, 4)
    gal_pixel_coords_n2: (Ngal_max, 2)
    gal_mask_n:          (Ngal_max,)
    cube_shape_zyx:      (Z, Y, X)

    Returns
    -------
    sampled_cubes : np.ndarray, shape (Nseed, Z, Y, X)
    model_t_indices : np.ndarray, shape (T_inf,)
    alpha_bars_inf : np.ndarray, shape (T_inf,)
    """
    betas_inf, alphas_inf, alpha_bars_inf, model_t_indices = \
        make_respaced_inference_schedule(alpha_bars_train, num_inference_steps)

    cond_batch = jnp.asarray(cond_image_hwc[None, ...], dtype=jnp.float32)
    gal_features_batch = jnp.asarray(gal_features_n4[None, ...], dtype=jnp.float32)
    gal_pixel_coords_batch = jnp.asarray(gal_pixel_coords_n2[None, ...], dtype=jnp.float32)
    gal_mask_batch = jnp.asarray(gal_mask_n[None, ...], dtype=jnp.float32)

    sample_shape = (1, cube_shape_zyx[0], cube_shape_zyx[1], cube_shape_zyx[2], 1)

    samples = []

    for seed in seeds:
        print(f"    Sampling seed={int(seed)}")
        rng_key = jax.random.PRNGKey(int(seed))

        sampled = sample_ddpm_respaced(
            model_apply=model.apply,
            params=params,
            cond_images=cond_batch,
            gal_features=gal_features_batch,
            gal_pixel_coords=gal_pixel_coords_batch,
            gal_mask=gal_mask_batch,
            sample_shape=sample_shape,
            rng_key=rng_key,
            betas_inf=betas_inf,
            alphas_inf=alphas_inf,
            alpha_bars_inf=alpha_bars_inf,
            model_t_indices=model_t_indices,
        )

        sampled = np.array(sampled[0, ..., 0], dtype=np.float32)  # (Z, Y, X)
        samples.append(sampled)

    sampled_cubes = np.stack(samples, axis=0)  # (Nseed, Z, Y, X)

    return (
        sampled_cubes,
        np.array(model_t_indices, dtype=np.int32),
        np.array(alpha_bars_inf, dtype=np.float32),
    )


for test_idx in test_indices:
    print("\n" + "=" * 80)
    print(f"Processing validation example test_idx = {test_idx}")

    imgs = images_all[test_idx]                              # (C, H, W)
    imgs_hwc = np.transpose(imgs, (1, 2, 0))                 # (H, W, C)
    true_cube = cubes_all[test_idx]                          # (Z, Y, X)

    gal_features = gal_features_all[test_idx]                # (max_nodes, 4)
    gal_pixel_coords = gal_pixel_coords_all[test_idx]        # (max_nodes, 2)
    gal_mask = mask_all[test_idx]                            # (max_nodes,)

    all_samples_this_example = []
    used_step_counts = []
    used_model_t_indices = []
    used_alpha_bars_inf = []

    for nsteps in num_inference_steps_list:
        print(f"\n  Using num_inference_steps = {nsteps}")

        sampled_cubes, model_t_idx, alpha_bars_inf = generate_cube_samples_for_schedule(
            model=model,
            params=params,
            cond_image_hwc=imgs_hwc,
            gal_features_n4=gal_features,
            gal_pixel_coords_n2=gal_pixel_coords,
            gal_mask_n=gal_mask,
            seeds=seeds,
            cube_shape_zyx=true_cube.shape,
            alpha_bars_train=alpha_bars_train,
            num_inference_steps=nsteps,
        )

        all_samples_this_example.append(sampled_cubes)
        used_step_counts.append(int(nsteps))
        used_model_t_indices.append(model_t_idx)
        used_alpha_bars_inf.append(alpha_bars_inf)

    all_samples_this_example = np.stack(all_samples_this_example, axis=0)
    # shape = (N_resolutions, Nseed, Z, Y, X)

    save_path = os.path.join(
        out_dir,
        f"sampled_cubes_validx_{test_idx:04d}{suffix}_respaced_multistep.npz"
    )

    model_t_indices_obj = np.empty(len(used_model_t_indices), dtype=object)
    alpha_bars_inf_obj = np.empty(len(used_alpha_bars_inf), dtype=object)

    for i in range(len(used_model_t_indices)):
        model_t_indices_obj[i] = used_model_t_indices[i]
        alpha_bars_inf_obj[i] = used_alpha_bars_inf[i]

    np.savez_compressed(
        save_path,
        test_idx=np.int32(test_idx),
        seeds=seeds.astype(np.int32),
        num_inference_steps=np.array(used_step_counts, dtype=np.int32),
        model_t_indices=model_t_indices_obj,
        alpha_bars_inference=alpha_bars_inf_obj,
        conditioning_images=imgs.astype(np.float32),              # (C, H, W)
        gal_features=gal_features.astype(np.float32),             # (max_nodes, 4)
        gal_pixel_coords=gal_pixel_coords.astype(np.float32),     # (max_nodes, 2)
        gal_mask=gal_mask.astype(np.float32),                     # (max_nodes,)
        true_cube=true_cube.astype(np.float32),                   # (Z, Y, X)
        sampled_cubes=all_samples_this_example.astype(np.float32),
    )

    print(f"\nSaved to:\n{save_path}")

print("\nDone.")