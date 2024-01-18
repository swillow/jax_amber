# jax_amber

Calculate the Amber Potential Energy using jax with the following features:

* Employ Ewald Summation for Coulomb Interaction

* Utilize OBC2 for implicit Solvent

* Incorporate Real NVP (real-valued non-volume preserving) model.

* Integrate Real NVP with VAE (Auto-Encoding with Variational Bayes).

* Note: Real NVP and (Real NVP+VAE) have a limitation - the minimization (optax.adam method) may be unable to overcome the potential energy barriers. To overcome the potential energy barriers, we conduct minimization (optax.adam) using pairwise distance map. (Here, pairwise refers to obtaining it from nonbonded pairwise interactions.)

* model_dmap.py : Train the end states using a pairwise distance map to eliminate the potential energy barriers between end states.

* model_dmap_vae.py : This follows the same scheme as model_dmap.py, but the neural network parameters in Real NVP are based on  VAE (Auto-Enconding with Variational Bayes).

* model_rnvp.py : Train the end states using the Amber potential energies (Boltzmann distribution). The primary reason for the failure in accurately estimating free energy with the Amber potential energies is the presence of numerous potential energy barriers between end states that the minimization process cannot overcome.

* model_vae.py : Similar to model_rnvp.py, but the neural network parameters in Real NVP are based on VAE (Auto-Enconding with Variational Bayes: https://github.com/google/flax/tree/main/examples/vae/). 

* HOWTO:
> python model_dmap.py test/input_dmap.json

> python model_rnvp.py test/input_rnvp.json

> python model_vae.py test/input_vae.json
