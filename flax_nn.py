from typing import Sequence

import jax 
import jax.numpy as jnp
from flax import linen as nn 
from flax.training import train_state 
from flax.linen.initializers import zeros as nn_zeros
from flax.linen.initializers import lecun_normal

default_kernel_init = lecun_normal()


"""
Reference code: Real NVP
TinyVolt/normalizing-flows
"""


class AfflineCoupling(nn.Module):
    input_size: int 
    i_dim: int 
    hidden_layers: int 
    hidden_dim : int 
    fixed_atoms: Sequence[int]

    @nn.compact
    def __call__ (self, inputs, reverse=False):

        fixed_mask = jnp.ones ((self.input_size), dtype=jnp.int32).reshape(-1,3)
        fixed_mask = fixed_mask.at[:,self.i_dim].set(0)
        moved_mask = jnp.int32(1) - fixed_mask
        moved_mask = moved_mask.at[self.fixed_atoms,self.i_dim].set(0)
        moved_mask = moved_mask.reshape (1,-1)
        fixed_mask = fixed_mask.reshape (1,-1)
        y = inputs*fixed_mask
        
        for _ in range (self.hidden_layers):
            y = nn.relu (nn.Dense (features=self.hidden_dim, kernel_init=default_kernel_init) (y))
    
        log_scale = nn.Dense (features=self.input_size, kernel_init=nn_zeros) (y)
        shift     = nn.Dense (features=self.input_size, kernel_init=nn_zeros) (y)
        shift     = shift*moved_mask 
        log_scale = log_scale*moved_mask
        
        if reverse:
            log_scale = -log_scale
            outputs = (inputs-shift)*jnp.exp(log_scale)
        else:
            outputs = inputs*jnp.exp(log_scale) + shift
      
        return outputs, log_scale 



class Encoder (nn.Module):
    latents: int 
    
    @nn.compact
    def __call__ (self, x):
        #d_hidden = [512, 256, 128, 64, 32, 16, 8]
        d_hidden = [256, 128, 64, 32, 16, 8]
        for h_dim in d_hidden:
            x = nn.relu(nn.Dense (h_dim) (x))
            
        mean_x = nn.Dense (self.latents, kernel_init=nn_zeros, name='fc5_mean') (x)
        logvar_x = nn.Dense (self.latents, name='fc5_logvar') (x)

        return mean_x, logvar_x 

    
class Decoder (nn.Module):
    out_dim: int 

    @nn.compact
    def __call__ (self, z):
        #d_hidden = [512, 256, 128, 64, 32, 16, 8]
        d_hidden = [256, 128, 64, 32, 16, 8]
        
        for h_dim in d_hidden[::-1]:
            z = nn.relu(nn.Dense (h_dim) (z) )
            
        log_scale = nn.Dense (self.out_dim, kernel_init=nn_zeros, name='fc5_scale') (z)
        shift = nn.Dense (self.out_dim, kernel_init=nn_zeros, name='fc5_shift') (z)

        return log_scale, shift
        


class VAE (nn.Module):
    input_size: int
    i_dim: int
    latents: int
    fixed_atoms: Sequence[int]

    def setup (self):
        self.encoder = Encoder (self.latents)
        self.decoder = Decoder (self.input_size)

    def __call__ (self, x, z_rng, reverse=False):
        fixed_mask = jnp.ones ((self.input_size), dtype=jnp.int32).reshape(-1,3)
        fixed_mask = fixed_mask.at[:,self.i_dim].set(0)
        moved_mask = jnp.int32(1) - fixed_mask
        moved_mask = moved_mask.at[self.fixed_atoms,self.i_dim].set(0)
        moved_mask = moved_mask.reshape (1,-1)
        fixed_mask = fixed_mask.reshape (1,-1)
        y = x*fixed_mask 

        z_mean, z_logvar = self.encoder (y)
        z = z_mean + \
            jax.random.normal(z_rng, z_logvar.shape)*jnp.exp(0.5*z_logvar)
        log_scale, shift = self.decoder (z)
        log_scale = log_scale*moved_mask 
        shift = shift*moved_mask 

        if reverse:
            log_scale = -log_scale 
            recon_x = (x-shift)*jnp.exp(log_scale)
        else:
            recon_x = x*jnp.exp(log_scale) + shift 

        return recon_x, log_scale, z_mean, z_logvar 
    


class realNVP3 (nn.Module):
    input_size: int 
    hidden_layers: int 
    hidden_dim : int 
    fixed_atoms: Sequence[int]
    
    def setup (self):
        
        self.af_x = AfflineCoupling (self.input_size, i_dim=0, 
                                     hidden_layers=self.hidden_layers, 
                                     hidden_dim=self.hidden_dim,
                                     fixed_atoms=self.fixed_atoms)
        self.af_y = AfflineCoupling (self.input_size, i_dim=1, 
                                     hidden_layers=self.hidden_layers, 
                                     hidden_dim=self.hidden_dim,
                                     fixed_atoms=self.fixed_atoms)
        self.af_z = AfflineCoupling (self.input_size, i_dim=2, 
                                     hidden_layers=self.hidden_layers, 
                                     hidden_dim=self.hidden_dim,
                                     fixed_atoms=self.fixed_atoms)

    @nn.compact
    def __call__ (self, inputs, reverse=False):
        n_conf, n_atoms, n_dim = inputs.shape 
        
        outputs = inputs.reshape (n_conf, -1)
        if reverse:
            outputs, log_J_z = self.af_z (outputs, reverse)
            outputs, log_J_y = self.af_y (outputs, reverse)
            outputs, log_J_x = self.af_x (outputs, reverse)
        else:
            outputs, log_J_x = self.af_x (outputs)
            outputs, log_J_y = self.af_y (outputs)
            outputs, log_J_z = self.af_z (outputs)

        return outputs.reshape(n_conf, n_atoms, n_dim), \
                (log_J_x + log_J_y + log_J_z).sum(axis=-1)

    
class VAErealNVP3 (nn.Module):
    input_size: int
    latents: int
    fixed_atoms: Sequence[int]
    
    def setup (self):
        
        self.af_x = VAE (self.input_size, i_dim=0, 
                        latents=self.latents,
                        fixed_atoms=self.fixed_atoms)
        self.af_y = VAE (self.input_size, i_dim=1, 
                        latents=self.latents,
                        fixed_atoms=self.fixed_atoms)
        self.af_z = VAE (self.input_size, i_dim=2, 
                        latents=self.latents,
                        fixed_atoms=self.fixed_atoms)

    @nn.compact
    def __call__ (self, inputs, rng, reverse=False):
        n_conf, n_atoms, n_dim = inputs.shape 
        
        outputs = inputs.reshape (n_conf, -1)
        rng, rng_x, rng_y, rng_z = jax.random.split(rng, num=4)
        if reverse:
            outputs, log_J_z, z_mu, z_logvar = self.af_z (outputs, rng_z, reverse)
            outputs, log_J_y, y_mu, y_logvar = self.af_y (outputs, rng_y, reverse)
            outputs, log_J_x, x_mu, x_logvar = self.af_x (outputs, rng_z, reverse)
        else:
            outputs, log_J_x, x_mu, x_logvar = self.af_x (outputs, rng_x)
            outputs, log_J_y, y_mu, y_logvar = self.af_y (outputs, rng_y)
            outputs, log_J_z, z_mu, z_logvar = self.af_z (outputs, rng_z)

        return outputs.reshape(n_conf, n_atoms, n_dim), \
                (log_J_x + log_J_y + log_J_z).sum(axis=-1), \
                (x_mu, y_mu, z_mu), (x_logvar, y_logvar, z_logvar)

