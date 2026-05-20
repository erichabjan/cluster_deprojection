import os
import sys
import time
import pickle
from typing import Dict, List

import h5py
import numpy as np

import jax
import jax.numpy as jnp
from flax.core import freeze

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cnn_mlp_globals_model import CNNSetGlobalsModel, GlobalsModelConfig


DATA_PATH = "/projects/mccleary_group/habjan.e/TNG/Data/conditional_diffusion_data/"
TRAIN_FILE = "cond_diffusion_16cubed_16img_train.h5"
VAL_FILE = "cond_diffusion_16cubed_16img_test.h5"

PARAMS_PATH = (
    "/home/habjan.e/TNG/cluster_deprojection/cnn_mlp_globals/"
    "cnn_mlp_globals_models/cnn_mlp_globals_params_16img_v1.pkl"
)

OUT_DIR = (
    "/projects/mccleary_group/habjan.e/TNG/Data/"
    "conditional_diffusion_data/cnn_mlp_globals"
)

BATCH_SIZE = 64


def load_split_to_memory(file_path: str) -> Dict[str, object]:
    print(f"\nPreloading {file_path} into memory...")
    start = time.time()

    with h5py.File(file_path, "r") as f:
        sample_ids: List[str] = sorted(f.keys())
        n_samples = len(sample_ids)
        first = f[sample_ids[0]]

        images_shape = first["images"].shape
        feat_shape = first["gal_features"].shape
        pix_shape = first["gal_pixel_coords"].shape
        mask_shape = first["mask"].shape
        targ_shape = first["globals_target"].shape

        images = np.zeros((n_samples, *images_shape), dtype=np.float32)
        gal_features = np.zeros((n_samples, *feat_shape), dtype=np.float32)
        gal_pixel_coords = np.zeros((n_samples, *pix_shape), dtype=np.float32)
        mask = np.zeros((n_samples, *mask_shape), dtype=np.float32)
        targets = np.zeros((n_samples, *targ_shape), dtype=np.float32)

        sim_labels: List[str] = []
        cluster_indices = np.zeros(n_samples, dtype=np.int64)
        cluster_masses = np.zeros(n_samples, dtype=np.float64)
        ids = np.zeros(n_samples, dtype=np.int64)

        for i, sid in enumerate(sample_ids):
            g = f[sid]
            images[i] = g["images"][:]
            gal_features[i] = g["gal_features"][:]
            gal_pixel_coords[i] = g["gal_pixel_coords"][:]
            mask[i] = g["mask"][:]
            targets[i] = g["globals_target"][:]

            sim = g.attrs.get("simulation", b"")
            if isinstance(sim, bytes):
                sim = sim.decode("utf-8")
            sim_labels.append(str(sim))
            cluster_indices[i] = int(g.attrs.get("cluster_index", -1))
            cluster_masses[i] = float(g.attrs.get("cluster_mass", np.nan))
            ids[i] = int(g.attrs.get("id", -1))

        file_attrs = {}
        for k in f.attrs.keys():
            v = f.attrs[k]
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            file_attrs[k] = v

    elapsed = time.time() - start
    print(f"  Loaded {n_samples} samples in {elapsed:.2f}s")

    return dict(
        sample_ids=sample_ids,
        images=images,
        gal_features=gal_features,
        gal_pixel_coords=gal_pixel_coords,
        mask=mask,
        targets=targets,
        sim_labels=sim_labels,
        cluster_indices=cluster_indices,
        cluster_masses=cluster_masses,
        ids=ids,
        file_attrs=file_attrs,
    )


def make_predict_fn(model, params):
    @jax.jit
    def predict(images, gal_features, gal_pixel_coords, mask):
        return model.apply(
            {"params": params},
            cond_images=images,
            gal_features=gal_features,
            gal_pixel_coords=gal_pixel_coords,
            gal_mask=mask,
        )

    return predict


def predict_split(predict_fn, data: Dict[str, object], batch_size: int) -> np.ndarray:
    n = data["images"].shape[0]
    preds = np.zeros((n, data["targets"].shape[1]), dtype=np.float32)

    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)

        images = data["images"][i:j]                       # (B,C,H,W)
        images = np.transpose(images, (0, 2, 3, 1))        # (B,H,W,C)
        gal_features = data["gal_features"][i:j]
        gal_pixel_coords = data["gal_pixel_coords"][i:j]
        mask = data["mask"][i:j]

        out = predict_fn(
            jnp.asarray(images),
            jnp.asarray(gal_features),
            jnp.asarray(gal_pixel_coords),
            jnp.asarray(mask),
        )
        out.block_until_ready()
        preds[i:j] = np.asarray(out)

    return preds


def destandardize(pred_std: np.ndarray, attrs: Dict[str, object]) -> np.ndarray:
    """
    globals_target_columns = [mass_log10_msun, axis_a_mpc, axis_b_mpc, axis_c_mpc]

    mass_log10_msun standardized using cube_mass_log10_mean / cube_mass_log10_std.
    axis_{a,b,c}_mpc standardized using axis_mean[i] / axis_std[i].
    """
    mass_mean = float(attrs["cube_mass_log10_mean"])
    mass_std = float(attrs["cube_mass_log10_std"])
    axis_mean = np.asarray(attrs["axis_mean"], dtype=np.float32).reshape(-1)
    axis_std = np.asarray(attrs["axis_std"], dtype=np.float32).reshape(-1)

    out = np.zeros_like(pred_std, dtype=np.float32)
    out[:, 0] = pred_std[:, 0] * mass_std + mass_mean
    out[:, 1:4] = pred_std[:, 1:4] * axis_std[None, :] + axis_mean[None, :]
    return out


def save_predictions(out_path: str, data: Dict[str, object], preds_std: np.ndarray):
    targets_std = data["targets"].astype(np.float32)
    attrs = data["file_attrs"]
    preds_phys = destandardize(preds_std, attrs)
    targets_phys = destandardize(targets_std, attrs)

    sample_ids_bytes = np.asarray(
        [s.encode("utf-8") for s in data["sample_ids"]], dtype="S16"
    )
    sim_labels_bytes = np.asarray(
        [s.encode("utf-8") for s in data["sim_labels"]], dtype="S32"
    )

    column_names = np.asarray(
        [b"mass_log10_msun", b"axis_a_mpc", b"axis_b_mpc", b"axis_c_mpc"], dtype="S32"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("sample_ids", data=sample_ids_bytes)
        f.create_dataset("simulation", data=sim_labels_bytes)
        f.create_dataset("cluster_index", data=data["cluster_indices"])
        f.create_dataset("cluster_mass", data=data["cluster_masses"])
        f.create_dataset("id", data=data["ids"])

        f.create_dataset("predictions_standardized", data=preds_std)
        f.create_dataset("predictions_physical", data=preds_phys)
        f.create_dataset("targets_standardized", data=targets_std)
        f.create_dataset("targets_physical", data=targets_phys)

        f.create_dataset("column_names", data=column_names)

        f.attrs["cube_mass_log10_mean"] = float(attrs["cube_mass_log10_mean"])
        f.attrs["cube_mass_log10_std"] = float(attrs["cube_mass_log10_std"])
        f.attrs["axis_mean"] = np.asarray(attrs["axis_mean"], dtype=np.float64)
        f.attrs["axis_std"] = np.asarray(attrs["axis_std"], dtype=np.float64)
        f.attrs["params_path"] = PARAMS_PATH
        f.attrs["source_file"] = data["source_file"]

    print(f"  Wrote {out_path}")


def build_model() -> CNNSetGlobalsModel:
    cfg = GlobalsModelConfig(
        base_channels=32,
        channel_mults=(1, 2, 4),
        galaxy_token_dim=128,
        num_attention_heads=4,
        num_global_queries=8,
        coord_image_size=16,
        head_hidden=(256, 256),
        out_dim=4,
    )
    return CNNSetGlobalsModel(cfg=cfg)


def main():
    print("JAX devices:", jax.devices())

    print(f"\nLoading params from {PARAMS_PATH}")
    with open(PARAMS_PATH, "rb") as f:
        params = pickle.load(f)
    params = freeze(params)

    model = build_model()
    predict_fn = make_predict_fn(model, params)

    splits = [
        ("train", os.path.join(DATA_PATH, TRAIN_FILE),
         os.path.join(OUT_DIR, "cnn_mlp_globals_predictions_train.h5")),
        ("val", os.path.join(DATA_PATH, VAL_FILE),
         os.path.join(OUT_DIR, "cnn_mlp_globals_predictions_val.h5")),
    ]

    for split_name, in_path, out_path in splits:
        print(f"\n=== Split: {split_name} ===")
        data = load_split_to_memory(in_path)
        data["source_file"] = in_path

        t0 = time.time()
        preds_std = predict_split(predict_fn, data, BATCH_SIZE)
        print(f"  Inference time: {time.time() - t0:.2f}s for {preds_std.shape[0]} samples")

        save_predictions(out_path, data, preds_std)


if __name__ == "__main__":
    main()
