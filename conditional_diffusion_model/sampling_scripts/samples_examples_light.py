import os
import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp

### Path of archived models
sys.path.append(os.getcwd())
sys.path.append('/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/cond_diff_old_models')
sys.path.append('/home/habjan.e/TNG/cluster_deprojection/conditional_diffusion_model')

### Import archived model
from conditional_diffusion_3d_model import ConditionalUNet3D, DiffusionModelConfig
from train_conditional_diffusion import preload_hdf5_to_memory, sample_ddpm

suffix = "_16cube_16img_v9"

val_path = "/projects/mccleary_group/habjan.e/TNG/Data/conditional_diffusion_data/cond_diffusion_16cubed_16img_test.h5"

model_dir = "/home/habjan.e/TNG/cluster_deprojection/conditional_diffusion_model/conditional_diffusion_models"
param_path = os.path.join(model_dir, f"cond_diffusion_params{suffix}.pkl")
sched_path = os.path.join(model_dir, f"cond_diffusion_schedule{suffix}.pkl")

out_dir = "/projects/mccleary_group/habjan.e/TNG/Data/conditional_diffusion_data/cd_examples"
os.makedirs(out_dir, exist_ok=True)


### Number of examples and seed
num_examples = 50
fixed_seed = 0
selection_seed = 12345


### Load data
data_dict = preload_hdf5_to_memory(val_path)

images_all = data_dict["images"]   # (N, C, H, W)
cubes_all  = data_dict["cubes"]    # (N, Z, Y, X)
gal_features_all = data_dict["gal_features"]        # (N, max_nodes, 4)
gal_pixel_coords_all = data_dict["gal_pixel_coords"]# (N, max_nodes, 2)
mask_all = data_dict["mask"]

n_test = images_all.shape[0]
print(f"Loaded test set with {n_test} examples.")

if num_examples > n_test:
    raise ValueError(f"Requested num_examples={num_examples}, but test set only has {n_test} examples.")

# Randomly choose distinct test examples
rng = np.random.default_rng(selection_seed)
test_indices = np.sort(rng.choice(n_test, size=num_examples, replace=False)).astype(np.int32)

print(f"Sampling {len(test_indices)} random test examples with fixed_seed={fixed_seed}")
print("Selected test indices:")
print(test_indices)


### Load model
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
    galaxy_token_dim=128,
    num_attention_heads=4,
    coord_image_size=16,
)

model = ConditionalUNet3D(cfg=cfg)


### Prepare full batch of conditioning data
images_batch = images_all[test_indices]                         # (B, C, H, W)
images_batch = np.transpose(images_batch, (0, 2, 3, 1))        # (B, H, W, C)
true_cubes = cubes_all[test_indices]                            # (B, Z, Y, X)
gal_features_batch = gal_features_all[test_indices]             # (B, max_nodes, 4)
gal_pixel_coords_batch = gal_pixel_coords_all[test_indices]     # (B, max_nodes, 2)
gal_mask_batch = mask_all[test_indices]                         # (B, max_nodes)

B = len(test_indices)
cube_shape = true_cubes.shape[1:]  # (Z, Y, X)
sample_shape = (B, cube_shape[0], cube_shape[1], cube_shape[2], 1)

print(f"\nSampling {B} cubes as a single batch ...")
print(f"  sample_shape = {sample_shape}")

rng_key = jax.random.PRNGKey(int(fixed_seed))

sampled = sample_ddpm(
    model_apply=model.apply,
    params=params,
    cond_images=jnp.asarray(images_batch, dtype=jnp.float32),
    gal_features=jnp.asarray(gal_features_batch, dtype=jnp.float32),
    gal_pixel_coords=jnp.asarray(gal_pixel_coords_batch, dtype=jnp.float32),
    mask=jnp.asarray(gal_mask_batch, dtype=jnp.float32),
    sample_shape=sample_shape,
    rng_key=rng_key,
    betas=betas,
    alphas=alphas,
    alpha_bars=alpha_bars,
)

sampled_cubes = np.array(sampled[..., 0], dtype=np.float32)  # (B, Z, Y, X)
print("Sampling complete.")

# Store conditioning images in (C, H, W) format for consistency with original script
conditioning_images = images_all[test_indices]                  # (B, C, H, W)
gal_features_arr = gal_features_all[test_indices]               # (B, max_nodes, 4)
gal_pixel_coords_arr = gal_pixel_coords_all[test_indices]       # (B, max_nodes, 2)
gal_mask_arr = mask_all[test_indices]                           # (B, max_nodes)


save_path = os.path.join(
    out_dir,
    f"sampled_cubes_{num_examples}random_fixedseed_{fixed_seed}{suffix}.npz"
)

np.savez_compressed(
    save_path,
    test_indices=test_indices,
    fixed_seed=np.int32(fixed_seed),
    selection_seed=np.int32(selection_seed),
    conditioning_images=conditioning_images.astype(np.float32),   # (N,C,H,W)
    gal_features=gal_features_arr.astype(np.float32),             # (N,max_nodes,4)
    gal_pixel_coords=gal_pixel_coords_arr.astype(np.float32),     # (N,max_nodes,2)
    gal_mask=gal_mask_arr.astype(np.float32),                     # (N,max_nodes)
    true_cubes=true_cubes.astype(np.float32),                     # (N,Z,Y,X)
    sampled_cubes=sampled_cubes.astype(np.float32),               # (N,Z,Y,X)
)

print(f"\nSaved sampled cubes to:\n{save_path}")
print("Saved arrays:")
print(f"  test_indices.shape         = {test_indices.shape}")
print(f"  conditioning_images.shape  = {conditioning_images.shape}")
print(f"  gal_features.shape         = {gal_features_arr.shape}")
print(f"  gal_pixel_coords.shape     = {gal_pixel_coords_arr.shape}")
print(f"  gal_mask.shape             = {gal_mask_arr.shape}")
print(f"  true_cubes.shape           = {true_cubes.shape}")
print(f"  sampled_cubes.shape        = {sampled_cubes.shape}")
