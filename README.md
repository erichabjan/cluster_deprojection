The Graph Neural Networks (GNNs) developed in this repository are trained using the [BAHAMAS simulation](https://arxiv.org/abs/1603.02702) and are evaluated on the [TNG simulation](https://www.tng-project.org/data/). The code in the `deterministic_model` trains a GNN using a standard, point estimation architecture while the `probabilistic_model` folder trains a model using conditional flow matching (CFM). The models take the ($x$, $y$, $v_z$) coordinates of cluster-member galaxies and outputs the ($z$, $v_x$, $v_y$) for all cluster-member galaxies. The training and validation curves of the current point estimatint model are shown here: 

<p align="center">
  <img src="/figures/loss_curves.png" alt="loss_curves" width="75%" />
</p>

And the ability of this point estimating model to reconstruct the ($z$, $v_x$, $v_y$) coordiantes of single TNG cluster:

<p align="center">
  <img src="/figures/tng_predictions.png" alt="tng_predictions" width="75%" />
</p>

For the CFM model, we are able to obtain a posterior distribution for each reconstructed coordinates ($z$, $v_x$, $v_y$). Since it is expected that there are major degeneracies in the possible values of these unobserved coordinates, it is viatal that we quantify this degeneracy; in this case, a CFM allows for this degeneracy to be quantified with a PDF. Here we show the sampled posterior of the CFM for a simulated cluster from BAHAMAS:

<p align="center">
  <img src="/figures/cfm_predictions_z.png" width="25%" />
  <img src="/figures/cfm_predictions_vx.png" width="25%" />
  <img src="/figures/cfm_predictions_vy.png" width="25%" />
</p>

The reconstruction of these quantities will allow for the dynamical state of the clsuter to be probed and for projection effects to be better accounted for in both dynamical and weak lensing derived masses.