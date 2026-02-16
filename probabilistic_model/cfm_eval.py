import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jraph

from cfm_training_structure import preload_hdf5_to_memory, data_loader, sample_cfm
from cfm_gnn import GraphConvNet, CFMGraphModel

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Pagella", "Book Antiqua", "Palatino Linotype", "DejaVu Serif"]
})


def load_cfm_model(params_path, hidden_size=1024, num_mlp_layers=3, latent_size=128, target_dim=3):
    """Recreate model architecture exactly + load params."""
    backbone = GraphConvNet(
        latent_size=latent_size,
        hidden_size=hidden_size,
        num_mlp_layers=num_mlp_layers,
        message_passing_steps=5,
        skip_connections=True,
        edge_skip_connections=True,
        norm="none",
        attention=True,
        shared_weights=True,
        relative_updates=False,
        output_dim=target_dim,
        dropout_rate=0.0,
    )
    model = CFMGraphModel(backbone=backbone, target_dim=target_dim, time_emb_dim=32)

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    return model, params


def get_nth_batch(test_data, batch_size=1, n=0, shuffle=False):
    """Grab the n-th batch from the deterministic loader."""
    it = data_loader(test_data, batch_size=batch_size, shuffle=shuffle)
    for _ in range(n):
        next(it)
    return next(it)  # graph, tgt, mask


def posterior_samples_for_graph(
    params,
    model,
    graph,
    num_samples=32,
    ode_steps=64,
    target_dim=3,
    seed=0,
):
    rng = jax.random.PRNGKey(seed)
    samples = []
    for k in range(num_samples):
        rng, key = jax.random.split(rng)
        x = sample_cfm(
            params=params,
            apply_fn=model.apply,
            graph=graph,
            rng_key=key,
            num_steps=ode_steps,
            target_dim=target_dim,
        )
        samples.append(x)
    return jnp.stack(samples, axis=0)


def posterior_stats(samples, mask=None):
    """
    samples: (S, N, D)
    mask: (N,) optional 0/1 or bool
    returns dict of mean/std and quantiles
    """
    mean = jnp.mean(samples, axis=0)
    std  = jnp.std(samples, axis=0)

    q16 = jnp.quantile(samples, 0.16, axis=0)
    q50 = jnp.quantile(samples, 0.50, axis=0)
    q84 = jnp.quantile(samples, 0.84, axis=0)

    out = dict(mean=mean, std=std, q16=q16, q50=q50, q84=q84)

    if mask is not None:
        m = mask.astype(bool).reshape(-1)
        out = {k: v[m] for k, v in out.items()}
    return out


def masked(arr, mask):
    m = mask.astype(bool).reshape(-1)
    n = min(arr.shape[0], m.shape[0])
    return arr[:n][m[:n]]


def violin_vs_truth_density(samples, tgt, output_dir, mask=None, dim=0, label="vx",
                            max_nodes=200, seed=0, width=0.06, alpha=0.75,
                            bins=60, y_stat="median"):
    """
    Color each violin by local 2D density of points in (truth, posterior_stat) space.
      x_i = truth_i
      y_i = median(samples_i)  (or mean)

    samples: (S, N, D)
    tgt:     (N, D)
    mask:    (N,) optional
    """
    samples = np.asarray(samples)
    tgt = np.asarray(tgt)

    # align lengths
    N = min(tgt.shape[0], samples.shape[1])
    tgt = tgt[:N]
    samples = samples[:, :N, :]

    idx = np.arange(N)
    if mask is not None:
        m = np.asarray(mask).astype(bool).reshape(-1)[:N]
        idx = np.where(m)[0]

    # subsample nodes for readability
    rng = np.random.default_rng(seed)
    if len(idx) > max_nodes:
        idx = rng.choice(idx, size=max_nodes, replace=False)

    x_true = tgt[idx, dim]
    s = samples[:, idx, dim]  # (S, M)

    if y_stat == "median":
        y_rep = np.median(s, axis=0)
    elif y_stat == "mean":
        y_rep = np.mean(s, axis=0)
    else:
        raise ValueError("y_stat must be 'median' or 'mean'")

    # --- 2D density via histogram2d ---
    # density per bin (counts normalized by bin area)
    H, xedges, yedges = np.histogram2d(x_true, y_rep, bins=bins, density=True)

    # map each point (x_true[i], y_rep[i]) -> its bin density
    xi = np.clip(np.searchsorted(xedges, x_true, side="right") - 1, 0, H.shape[0]-1)
    yi = np.clip(np.searchsorted(yedges, y_rep,  side="right") - 1, 0, H.shape[1]-1)
    dens = H[xi, yi]

    # sort by x for nicer look
    order = np.argsort(x_true)
    x_true = x_true[order]
    y_rep  = y_rep[order]
    dens   = dens[order]
    data   = [samples[:, idx[o], dim] for o in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(
        dataset=data,
        positions=x_true,
        widths=width,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    if "cmedians" in parts:
        parts["cmedians"].set_linewidth(1.0)
    
    for pc in parts["bodies"]:
        pc.set_facecolor("k")
        pc.set_edgecolor("none")
        pc.set_alpha(alpha)

    # y=x line for reference
    lo = min(np.min(x_true), np.min([d.min() for d in data]))
    hi = max(np.max(x_true), np.max([d.max() for d in data]))
    ax.plot([lo * 1.5, hi * 1.5], [lo * 1.5, hi * 1.5], lw=3, c='k')
    ax.set_xlim(lo * 1.1, hi * 1.1)
    ax.set_ylim(lo * 1.1 , hi * 1.1)

    ax.set_xlabel(f"BAHAMAS {label}")
    ax.set_ylabel(f"Posterior Sample {label}")

    plt.tight_layout()
    fig.savefig(output_dir, bbox_inches="tight")


# --- paths ---
data_path = "/projects/mccleary_group/habjan.e/TNG/Data/GNN_SBI_data/"
test_file = "GNN_data_test.h5"

# --- load data ---
test_data = preload_hdf5_to_memory(data_path, test_file)

# path to the params you saved in train_cfm_model.py
params_path = os.path.join("/home/habjan.e/TNG/cluster_deprojection/probabilistic_model/CFM_models", "cfm_model_params_cfm_testing.pkl")  # adjust suffix

# --- load model/params ---
model, params = load_cfm_model(
    params_path,
    hidden_size=1024,
    num_mlp_layers=3,
    latent_size=128,
    target_dim=3,
)

# --- get a single test graph ---
graph, tgt, mask = get_nth_batch(test_data, batch_size=1, n=0, shuffle=False)

print("graph nodes:", graph.nodes.shape, "targets:", tgt.shape, "mask:", mask.shape)

# --- posterior sampling ---
samples = posterior_samples_for_graph(
    params=params,
    model=model,
    graph=graph,
    num_samples=128,
    ode_steps=128,
    target_dim=3,
    seed=0,
)

max_nodes = 25
width = 0.25

output_z = "/home/habjan.e/TNG/cluster_deprojection/figures/cfm_predictions_z_test.png"
violin_vs_truth_density(samples, tgt, output_z, mask=mask, dim=0, label=r"$z$-position",
                        max_nodes=max_nodes, width=width, alpha=0.3, bins=60, y_stat="median")

output_vx = "/home/habjan.e/TNG/cluster_deprojection/figures/cfm_predictions_vx_test.png"
violin_vs_truth_density(samples, tgt, output_vx, mask=mask, dim=1, label=r"$v_{x}$",
                        max_nodes=max_nodes, width=width, alpha=0.3, bins=60, y_stat="median")

output_vy = "/home/habjan.e/TNG/cluster_deprojection/figures/cfm_predictions_vy_test.png"
violin_vs_truth_density(samples, tgt, output_vy, mask=mask, dim=2, label=r"$v_{y}$",
                        max_nodes=max_nodes, width=width, alpha=0.3, bins=60, y_stat="median")

print('Plots saved!')