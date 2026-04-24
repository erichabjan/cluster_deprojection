In this repository, simulation particle data from the hydrodynamical simulations [BAHAMAS (BAryons and HAloes of MAssive Systems)](https://arxiv.org/abs/1603.02702) and [IllustriusTNG](https://www.tng-project.org/data/) are used as training/validation and test data, respectively, to train a conditional diffusion model to predict the three-dimensional mass density of a galaxy cluster. The three-dimensional cubes are constructed using a 10 Mpc<sup>3</sup> cube that is centered at the center-of-mass of the Friends-of-Friends halos found in each simulation. All simulation particles (i.e., dark matter, gas, stars, black holes) are added to their respective voxel inside of the cube. The conditional diffusion model learns to noise this density cube conditional on a set of xy-plane images (see the figure below) and observed cluster-member galaxy dynamics ($x$, $y$, $v_z$). The conditional images are fed into convolutional layers that lift the 2D channels into a 3D volume with feature vectors at each voxel. The dynamical information is also equipped with xy-image coordinates such that the conditional images can be queried at these coordinates; the ($x$, $y$, $v_z$) and queried coordinates are input to a multi-layered perceptron. Feature vectors for each galaxy are obtained and are cross-attended with a learned voxelized cube, allowing three-dimensional information from the cluster-dynamics to be gridded. This cross-attended cube, as well as the lifted three-dimensional cubes are fused and passed to a U-net. Lastly, each channel from the U-net are fused to produce a three-dimensional mass density cube at a given time step in the diffusion process. 

Here are the xy-plane conditioning images of a BAHAMAS valdiation example are shown: 

<p align="center">
  <img src="figures/conddiff_data.png" alt="cond_diff_data" width="95%" />
</p>

These xy conditioning images are paired with the cluster-member galaxy dynamics allowing for the non-parameteric mass distribution of a galaxy cluster to be inferred. The predicted total mass enclosed within this 10 Mpc<sup>3</sup> cube is compared with the true enclosed cube mass: 

<p align="center">
  <img src="figures/cube_masses.png" alt="cube_masses" width="50%" />
</p>

The best model at the moment is able to reconstruct cube masses to with a median difference $<1$% and a standard deviation of ~5%. By enabling three-dimensional, non-parametric cluster mass reconstruction across redshift and observational resolution, this framework opens the door to new cosmological tests of dark matter physics and the growth of structure across cosmic time. 