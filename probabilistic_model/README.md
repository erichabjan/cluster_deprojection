The CFM GNN model trained in this folder takes the ($x$, $y$, $v_z$) coordinates of cluster-member galaxies and outputs the ($z$, $v_x$, $v_y$) for all cluster-member galaxies. The posterior distribution for each reconstructed coordinates ($z$, $v_x$, $v_y$) is obtained by sampling the predicted velocity field. Since it is expected that there are major degeneracies in the possible values of these unobserved coordinates, it is viatal that we quantify this degeneracy; in this case, a CFM allows for this degeneracy to be quantified with a PDF. Here we show the sampled posterior of the CFM for a simulated cluster from BAHAMAS:

<p align="center">
  <img src="/figures/cfm_predictions_z.png" width="25%" />
  <img src="/figures/cfm_predictions_vx.png" width="25%" />
  <img src="/figures/cfm_predictions_vy.png" width="25%" />
</p>

The reconstruction of these quantities will allow for the dynamical state of the clsuter to be probed and for projection effects to be better accounted for in both dynamical and weak lensing derived masses.