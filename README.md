# jax_amber

Estimate the Amber Potential Energy using jax with the following features

* Ewald Summation for Coulomb Interaction

* OBC2 for implicit Solvent

* Real NVP (real-valued non-volume preserving) model is added.

* Real NVP with VAE (Auto-Encoding with Variational Bayes)  is added.

* Note: Real NVP and (Real NVP+VAE) have the limitation that the minimization (optax.adam method) cannot overcome the potential energy barriers. Hance, to overcome the potential energy barriers, we perform the minimization (optax.adam) based on pairwise distance map. (Here, pairwise is obtained from nonbonded pairwise.)

* model_dmap.py : Train the end states using pairwise distance map in order to remove the potential energy barriers between end states.

* model_dmap_vae.py : This is the same scheme as model_dmap.py. But the neural network parameters in real NVP are based on  VAE (Auto-Enconding with Variational Bayes).

* model_rnvp.py : Train the end states using the Amber potential energies (Boltzmann distribution). The key reason of the failure of the correct free energy estimation with the Amber potential energies is many potential energy barriers between end states that the minimization cannot cross over.

* model_vae.py : This is the same scheme as model_rnvp.py. But the neural network parameters in real NVP are based on VAE (Auto-Enconding with Variational Bayes). 

* HOWTO:
> python model_dmap.py test/input_dmap.py
> python model_rnvp.py test/input_rnvp.py
> python model_vae.py test/input_vae.py
