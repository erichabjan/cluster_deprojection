The Graph Neural Networks (GNN) developed in this repository are trained using the [BAHAMAS simulation](https://arxiv.org/abs/1603.02702) and are evaluated on the [TNG simulation](https://www.tng-project.org/data/). The code in the `deterministic_model` trains a GNN using a standard, deterministic architecture while the `probabilistic_model` folder trains a model using normalizing flows. The models take the ($x$, $y$, $v_z$) coordinates of cluster-member galaxies and outputs the cluster-centric radii and magnitude of velocities for all cluster-member galaxies. The training and validation curves of the current determinisitc model are shown here: 

<p align="center">
  <img src="../figures/loss_curves.png" alt="loss_curves" width="width:75%" />
</p>

And the ability of the model to reconstruct the cluster-centric radii and magnitude of velocities:

<p align="center">
  <img src="../figures/tng_predictions.png" alt="tng_predictions" width="width:75%" />
</p>

The reconstruction of these quantities will allow for the dynamical state of the clsuter to be probed and for projection effects to be accounted for in both dynamical and weak lensing derived masses. To see how the model is progressing, check out this project on [Weights and Biases](https://wandb.ai/erichabjan-northeastern-university/phase-space-GNN).