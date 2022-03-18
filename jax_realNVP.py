import jax 
import jax.numpy as jnp
import numpy as np 
from jax.example_libraries import stax

RT = jnp.float32 (8.3144621E-3 * 300.0) #* omm.unit.kilojoules_per_mole # kJ/mol
beta = 1.0/RT

"""
Real NVP
TinyVolt/normalizing-flows
"""

def AffineCheckerboardTransform (left=False, mask_fixed=None):

    def init_fun (rng, input_dim, hidden_dim=64, **kwargs):

        output_dim = input_dim * 2 # scale and translate
        net_init, apply_fun = stax.serial (
            stax.Dense (hidden_dim, W_init=jax.nn.initializers.zeros, 
                                    b_init=jax.nn.initializers.zeros),
            stax.Relu, 
            stax.Dense (hidden_dim, W_init=jax.nn.initializers.zeros, 
                                    b_init=jax.nn.initializers.zeros),
            stax.Relu, 
            stax.Dense (hidden_dim, W_init=jax.nn.initializers.zeros, 
                                    b_init=jax.nn.initializers.zeros),
            stax.Relu, 
            stax.Dense (hidden_dim, W_init=jax.nn.initializers.zeros, 
                                    b_init=jax.nn.initializers.zeros),
            stax.Relu, 
            stax.Dense (hidden_dim, W_init=jax.nn.initializers.zeros, 
                                    b_init=jax.nn.initializers.zeros),
            stax.Relu, 
            stax.Dense (output_dim, W_init=jax.nn.initializers.zeros, 
                                    b_init=jax.nn.initializers.zeros)
        )

        out_shape, params = net_init (rng, (input_dim, ))

        # create mask
        mask = np.arange(input_dim).reshape(1,-1)
        if not left:
            mask += 1
        fixed_mask = (mask%2)
        if mask_fixed is not None:
            print ('mask_fixed', mask_fixed.shape)
            print ('fixed_mask', fixed_mask.shape)
            fixed_mask[np.array(mask_fixed)] = 1

        fixed_mask = jnp.array(fixed_mask)
        moved_mask = 1 - fixed_mask

        def direct_fun (params, inputs, **kwargs):
            # inputs (1, input_dim)
            x_masked = inputs*fixed_mask
            
            log_scale, shift = apply_fun (params, x_masked).split (2, axis=-1)
            shift = shift * moved_mask
            log_scale = log_scale*moved_mask

            # Y = W*X + B
            outputs = inputs*jnp.exp(log_scale) + shift

            return outputs, log_scale

        def inverse_fun (params, inputs, **kwargs):
            x_masked = inputs*fixed_mask

            log_scale, shift = apply_fun (params, x_masked).split (2, axis=-1)

            shift = shift*moved_mask
            log_scale = log_scale*moved_mask
            outputs = (inputs-shift)*jnp.exp(-log_scale)

            return outputs, log_scale
        

        return params, direct_fun, inverse_fun
    
    return init_fun 


def Serial (*init_funs):

    def init_fun (rng, input_dim, **kwargs):

        all_params, direct_funs, inverse_funs = [], [], []

        for init_local_fun in init_funs:
            rng, layer_rng = jax.random.split (rng)
            param, direct_fun, inverse_fun = init_local_fun (layer_rng,
                                                            input_dim)
            all_params.append (param)
            direct_funs.append (direct_fun)
            inverse_funs.append (inverse_fun)

        
        def feed_forward (params, apply_funs, inputs):
            log_det_jacobians = jnp.zeros (inputs.shape)

            for apply_fun, param in zip(apply_funs, params):
                inputs, log_det_jacobian = apply_fun (param, inputs, **kwargs)
                log_det_jacobians += log_det_jacobian
            
            return inputs, log_det_jacobians
        
        def direct_fun (params, inputs, **kwargs):
            return feed_forward (params, direct_funs, inputs)

        def inverse_fun (params, inputs, **kwargs):
            return feed_forward (reversed(params),
                                 reversed(inverse_funs), inputs)
        
        return all_params, direct_fun, inverse_fun 

    return init_fun 



def realNVP():

    def init_fun (rng, input_dim, mask_fixed=None):

        transform_fun = Serial (AffineCheckerboardTransform(False,mask_fixed),
                                AffineCheckerboardTransform(True, mask_fixed) )

        layer_rng, input_rng = jax.random.split(rng)
        params, trans_direct, trans_inverse = transform_fun (layer_rng, 
                                                            input_dim)

        def direct_fun (params, inputs, **kwargs):
            """
            input has (n_conf, input_dim), input_dim = n_atom*n_dim
            """                                              
            return trans_direct (params, inputs)

        def inverse_fun (params, inputs, **kwargs):

            return trans_inverse (params, inputs)
        
        return params, direct_fun, inverse_fun
    
    return init_fun 

