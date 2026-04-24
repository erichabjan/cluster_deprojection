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

# Randomly choose 100 distinct test examples
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


def generate_cube_sample(
    model,
    params,
    cond_image_hwc,
    gal_features_n4,
    gal_pixel_coords_n2,
    gal_mask_n,
    seed,
    cube_shape_zyx,
):
    """
    cond_image_hwc:      (H, W, C)
    gal_features_n4:     (N, 4)
    gal_pixel_coords_n2: (N, 2)
    gal_mask_n:          (N,)
    cube_shape_zyx:      (Z, Y, X)

    returns:
        sampled_cube: (Z, Y, X)
    """
    cond_batch = jnp.asarray(cond_image_hwc[None, ...], dtype=jnp.float32)       # (1,H,W,C)
    gal_features_batch = jnp.asarray(gal_features_n4[None, ...], dtype=jnp.float32)
    gal_pixel_coords_batch = jnp.asarray(gal_pixel_coords_n2[None, ...], dtype=jnp.float32)
    gal_mask_batch = jnp.asarray(gal_mask_n[None, ...], dtype=jnp.float32)

    sample_shape = (1, cube_shape_zyx[0], cube_shape_zyx[1], cube_shape_zyx[2], 1)

    print(f"Sampling seed={seed} ...")
    rng_key = jax.random.PRNGKey(int(seed))

    sampled = sample_ddpm(
        model_apply=model.apply,
        params=params,
        cond_images=cond_batch,
        gal_features=gal_features_batch,
        gal_pixel_coords=gal_pixel_coords_batch,
        mask=gal_mask_batch,
        sample_shape=sample_shape,
        rng_key=rng_key,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
    )

    sampled_cube = np.array(sampled[0, ..., 0], dtype=np.float32)   # (Z,Y,X)
    return sampled_cube


conditioning_images_list = []
gal_features_list = []
gal_pixel_coords_list = []
gal_mask_list = []
true_cubes_list = []
sampled_cubes_list = []

for i, test_idx in enumerate(test_indices):
    print(f"\n[{i+1}/{len(test_indices)}] Processing test_idx={test_idx}")

    imgs = images_all[test_idx]                       # (C, H, W)
    imgs_hwc = np.transpose(imgs, (1, 2, 0))         # (H, W, C)
    true_cube = cubes_all[test_idx]                  # (Z, Y, X)

    gal_features = gal_features_all[test_idx]        # (max_nodes, 4)
    gal_pixel_coords = gal_pixel_coords_all[test_idx]# (max_nodes, 2)
    gal_mask = mask_all[test_idx]                    # (max_nodes,)

    sampled_cube = generate_cube_sample(
        model=model,
        params=params,
        cond_image_hwc=imgs_hwc,
        gal_features_n4=gal_features,
        gal_pixel_coords_n2=gal_pixel_coords,
        gal_mask_n=gal_mask,
        seed=fixed_seed,
        cube_shape_zyx=true_cube.shape,
    )

    conditioning_images_list.append(imgs.astype(np.float32))               # (C,H,W)
    gal_features_list.append(gal_features.astype(np.float32))              # (N,4)
    gal_pixel_coords_list.append(gal_pixel_coords.astype(np.float32))      # (N,2)
    gal_mask_list.append(gal_mask.astype(np.float32))                      # (N,)
    true_cubes_list.append(true_cube.astype(np.float32))                   # (Z,Y,X)
    sampled_cubes_list.append(sampled_cube.astype(np.float32))             # (Z,Y,X)

conditioning_images = np.stack(conditioning_images_list, axis=0)   # (N,C,H,W)
gal_features_arr = np.stack(gal_features_list, axis=0)             # (N,max_nodes,4)
gal_pixel_coords_arr = np.stack(gal_pixel_coords_list, axis=0)     # (N,max_nodes,2)
gal_mask_arr = np.stack(gal_mask_list, axis=0)                     # (N,max_nodes)
true_cubes = np.stack(true_cubes_list, axis=0)                     # (N,Z,Y,X)
sampled_cubes = np.stack(sampled_cubes_list, axis=0)               # (N,Z,Y,X)


save_path = os.path.join(
    out_dir,
    f"sampled_cubes_{num_examples}random_fixedseed_{fixed_seed}{suffix}.npz"
)

np.savez_compressed(
    save_path,
    test_indices=test_indices,
    fixed_seed=np.int32(fixed_seed),
    selection_seed=np.int32(selection_seed),
    conditioning_images=conditioning_images,   # (N,C,H,W)
    gal_features=gal_features_arr,             # (N,max_nodes,4)
    gal_pixel_coords=gal_pixel_coords_arr,     # (N,max_nodes,2)
    gal_mask=gal_mask_arr,                     # (N,max_nodes)
    true_cubes=true_cubes,                     # (N,Z,Y,X)
    sampled_cubes=sampled_cubes,               # (N,Z,Y,X)
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