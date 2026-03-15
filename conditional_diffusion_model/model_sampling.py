import os
import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(os.getcwd())

from conditional_diffusion_3d_model import ConditionalUNet3D, DiffusionModelConfig
from train_conditional_diffusion import preload_hdf5_to_memory, sample_ddpm


val_path = "/projects/mccleary_group/habjan.e/TNG/Data/conditional_diffusion_data/cond_diffusion_test.h5"

model_dir = os.path.join(os.getcwd(), "conditional_diffusion_models")
param_path = os.path.join(model_dir, "cond_diffusion_params_64cube_v2.pkl")
sched_path = os.path.join(model_dir, "cond_diffusion_schedule_64cube_v2.pkl")

out_dir = "/projects/mccleary_group/habjan.e/TNG/Data/conditional_diffusion_data/cd_samples"
os.makedirs(out_dir, exist_ok=True)


data_dict = preload_hdf5_to_memory(val_path)

images_all = data_dict["images"]   # (N, C, H, W)
cubes_all  = data_dict["cubes"]    # (N, Z, Y, X)

# Pick one validation example
test_idx = 0

# conditioning images for one sample
imgs = images_all[test_idx]                     # (C, H, W)
imgs_hwc = np.transpose(imgs, (1, 2, 0))       # (H, W, C)

# true cube for one sample
true_cube = cubes_all[test_idx]                # (Z, Y, X)


with open(param_path, "rb") as f:
    params = pickle.load(f)

with open(sched_path, "rb") as f:
    diffusion_cfg = pickle.load(f)

betas = jnp.asarray(diffusion_cfg["betas"])
alphas = jnp.asarray(diffusion_cfg["alphas"])
alpha_bars = jnp.asarray(diffusion_cfg["alpha_bars"])

cfg = DiffusionModelConfig(
    base_channels=32,
    channel_mults=(1, 2, 4),
    time_emb_dim=128,
    out_channels=1,
)

model = ConditionalUNet3D(cfg=cfg)


def generate_cube_samples(model, params, cond_image_hwc, seeds, cube_shape_zyx):
    """
    cond_image_hwc: (H, W, C)
    cube_shape_zyx: (Z, Y, X)

    returns:
        sampled_cubes: (Nseed, Z, Y, X)
    """
    samples = []

    cond_batch = jnp.asarray(cond_image_hwc[None, ...], dtype=jnp.float32)   # (1,H,W,C)
    sample_shape = (1, cube_shape_zyx[0], cube_shape_zyx[1], cube_shape_zyx[2], 1)

    for seed in seeds:
        print(f"Sampling seed={seed} ...")
        rng_key = jax.random.PRNGKey(seed)

        sampled = sample_ddpm(
            model_apply=model.apply,
            params=params,
            cond_images=cond_batch,
            sample_shape=sample_shape,
            rng_key=rng_key,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
        )

        sampled = np.array(sampled[0, ..., 0], dtype=np.float32)   # (Z,Y,X)
        samples.append(sampled)

    return np.stack(samples, axis=0)   # (Nseed,Z,Y,X)


seeds = np.arange(0, 100, dtype=np.int32)

sampled_cubes = generate_cube_samples(
    model=model,
    params=params,
    cond_image_hwc=imgs_hwc,
    seeds=seeds,
    cube_shape_zyx=true_cube.shape,
)


save_path = os.path.join(out_dir, f"sampled_cubes_validx_{test_idx:04d}.npz")

np.savez_compressed(
    save_path,
    test_idx=np.int32(test_idx),
    seeds=seeds,
    conditioning_images=imgs.astype(np.float32),   # (C,H,W)
    true_cube=true_cube.astype(np.float32),        # (Z,Y,X)
    sampled_cubes=sampled_cubes.astype(np.float32) # (Nseed,Z,Y,X)
)

print(f"Saved sampled cubes to:\n{save_path}")