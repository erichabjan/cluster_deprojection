The combined CNN-MLP model in this folder takes the ($x$, $y$, $v_z$) coordinates of a single cluster-member galaxy, the projected mass map of the cluster, and distribution images of the discrete locations of the ($x$, $y$, $v_z$) coordinates as inputs and outputs the ($z$, $v_x$, $v_y$) for a single cluster-member galaxy. The data images are visualized here: 

<p align="center">
  <img src="/figures/cnnmlp_data.png" alt="loss_curves" width="100%" />
</p>

And the ability of this point estimating model to reconstruct the ($z$, $v_x$, $v_y$) coordiantes of a validation BAHAMAS cluster:

<p align="center">
  <img src="/figures/cnnmlp_predictions.png" alt="tng_predictions" width="75%" />
</p>

This model is still a work in progress. The reconstruction of these quantities will allow for the dynamical state of the clsuter to be robustly probed, 6D dynamical substructure identification to be carried out and for projection effects to be better accounted for in both dynamical and weak lensing derived masses.