The point-estimating GNN in thie folder takes the ($x$, $y$, $v_z$) coordinates of cluster-member galaxies and outputs the ($z$, $v_x$, $v_y$) for all cluster-member galaxies. The training and validation curves of the current point estimatint model are shown here: 

<p align="center">
  <img src="/figures/loss_curves.png" alt="loss_curves" width="75%" />
</p>

And the ability of this point estimating model to reconstruct the ($z$, $v_x$, $v_y$) coordiantes of single TNG cluster:

<p align="center">
  <img src="/figures/tng_predictions.png" alt="tng_predictions" width="75%" />
</p>

This model is still a work in progress. The reconstruction of these quantities will allow for the dynamical state of the clsuter to be robustly probed, 6D dynamical substructure identification to be carried out and for projection effects to be better accounted for in both dynamical and weak lensing derived masses.