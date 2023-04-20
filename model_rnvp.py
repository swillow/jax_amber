import jax 
import jax.numpy as jnp
 
import optax 
import pymbar 

from flax.training import train_state 

import jax_amber
import flax_nn
import sys 
from util import get_trajectory, write_traj, checkpoint_load, checkpoint_save


RT = jnp.float32(8.3144621E-3 * 300.0) 
beta = jnp.float32(1.0)/RT 
nm2ang = jnp.float32(10.0)
ang2nm = jnp.float32(0.1)



def ener_restraint_fun (R_fixed, R_fixed0, kval):
    # E = kval [(x-x0)^2 + (y-y0)^2 + (z-z0)^2]
    dR = R_fixed - R_fixed0
    # dR[n_fixed_atoms,3]
    # enr = kval * \sum_{i=fixed_atoms}\sum_{j=x,y,z} (R_ij-R_ij^0)^2
    enr = kval*jnp.einsum('i,i->', dR, dR)
    
    return enr



def get_energy_values (x, ener_funs, R0):
    ener_wHO_fun, ener_bond_fun = ener_funs 
    enr_bnd = jax.vmap(ener_bond_fun) (x)
    enr_wHO = jax.vmap(ener_wHO_fun, in_axes=(0,None)) (x, R0)
    
    return enr_bnd, enr_wHO



def print_progress (state, inputs, 
                    ener_funs, ener_ref0, 
                    fixed_iatom, fixed_R0, 
                    fout):
    x_A, x_B = inputs
    R0_A, R0_B = fixed_R0
    enr_wHO_A0, enr_wHO_B0,_,_ = ener_ref0

    m_B, log_J_F = state.apply_fn ({'params':state.params}, x_A)
    m_A, log_J_R = state.apply_fn ({'params':state.params}, x_B, reverse=True)

    enr_bond_A, enr_wHO_A = get_energy_values (m_A, ener_funs, R0_A)
    enr_bond_B, enr_wHO_B = get_energy_values (m_B, ener_funs, R0_B)
    
    dU_F = enr_wHO_B - enr_wHO_A0
    dU_R = enr_wHO_A - enr_wHO_B0 
    phi_F = beta*dU_F - log_J_F
    phi_R = beta*dU_R - log_J_R 

    Z_A = m_A[:,fixed_iatom,2].mean()
    Z_B = m_B[:,fixed_iatom,2].mean()

    print ('R_A R_B             {:12.6f} {:12.6f}'.format (R0_A[-1], R0_B[-1]), file=fout)
    print ('Fixed_Z             {:12.6f} {:12.6f}'.format(Z_A, Z_B), file=fout)
    print ('<-log_J>(kJ/mol)    {:12.6f} {:12.6f}'.format(RT*(-log_J_F-log_J_R).mean(), -RT*log_J_F.mean()), file=fout)
    print (' <U_bond>(kJ/mol)   {:12.6f} {:12.6f}'.format (enr_bond_A.mean(), enr_bond_B.mean()), file=fout)
    print (' <U_wHO>(kJ/mol)    {:12.6f} {:12.6f}'.format (enr_wHO_A.mean(), enr_wHO_B.mean()), file=fout)
    print ('<dU_wHO>(kJ/mol)    {:12.6f} {:12.6f}'.format (dU_R.mean(), dU_F.mean()), file=fout)
    print ('<phi_wHO>(kJ/mol)   {:12.6f} {:12.6f}'.format (RT*phi_R.mean(), RT*phi_F.mean()), file=fout) 
    # pymber 3
    #f_BAR_wHO = pymbar.BAR (phi_F, phi_R, 
    #                    relative_tolerance=1.0e-5,
    #                    verbose=False,
    #                    compute_uncertainty=False)
    # pymbar 4
    f_BAR_wHO = pymbar.bar (phi_F, phi_R,compute_uncertainty=False,relative_tolerance=1.0e-5)['Delta_f']

    print ('LBAR(kJ/mol)        {:12.6f}'.format ( RT*f_BAR_wHO ), file=fout)



def loss_value (ener_wHO_fn, ener_bond_fn,
                enr0_wHO, m_B, log_J_F, m_A, log_J_R, fixed_R0):

    enr_wHO_A0, enr_wHO_B0, enr_bnd_A0, enr_bnd_B0 = enr0_wHO
    R0_A, R0_B = fixed_R0

    enr_A = jax.vmap(ener_wHO_fn, in_axes=(0,None)) (m_A, R0_A)
    enr_B = jax.vmap(ener_wHO_fn, in_axes=(0,None)) (m_B, R0_B)
    enr_bnd_A = jax.vmap(ener_bond_fn) (m_A)
    enr_bnd_B = jax.vmap(ener_bond_fn) (m_B)

    loss_F = beta*(enr_B - enr_wHO_A0) - log_J_F 
    loss_R = beta*(enr_A - enr_wHO_B0) - log_J_R
    diff_bnd_A = beta*(enr_bnd_A.mean() - enr_bnd_A0)
    diff_bnd_B = beta*(enr_bnd_B.mean() - enr_bnd_B0)
    diff_bnd_A2 = diff_bnd_A**2
    diff_bnd_B2 = diff_bnd_B**2
    loss = loss_F.mean() + loss_R.mean() 
    loss_wBnd = loss + diff_bnd_A2 + diff_bnd_B2 
    return loss_wBnd, loss


def get_current_loss_values (state, inputs, inputs_test, 
                             ener_wHO_fun, ener_bond_fun, 
                             ener_ref0, ener_ref0_test, 
                             fixed_R0):

    x_A, x_B = inputs
    tx_A, tx_B = inputs_test 

    m_B, log_J_F = state.apply_fn ({'params':state.params}, tx_A)
    m_A, log_J_R = state.apply_fn ({'params':state.params}, tx_B, reverse=True)

    _, loss_test = loss_value (ener_wHO_fun, ener_bond_fun, ener_ref0_test,
                             m_B, log_J_F, m_A, log_J_R, fixed_R0)
    
    m_B, log_J_F = state.apply_fn ({'params':state.params}, x_A)
    m_A, log_J_R = state.apply_fn ({'params':state.params}, x_B, reverse=True)

    loss_wBnd, loss = loss_value (ener_wHO_fun, ener_bond_fun, ener_ref0,
                             m_B, log_J_F, m_A, log_J_R, fixed_R0)
    
    return loss_wBnd, loss, loss_test, (m_A, m_B)



def main_train (json_data):

    fout = open (json_data['fname_log'], 'w', 1)
    x_A, tx_A = get_trajectory (json_data['fname_prmtop'],
                          json_data['fname_dcd_A'],
                          json_data['nsamp'])
    x_B, tx_B = get_trajectory (json_data['fname_prmtop'],
                          json_data['fname_dcd_B'],
                          json_data['nsamp'])

    inputs = (x_A, x_B)
    inputs_test = (tx_A, tx_B)
    nconf = x_A.shape[0]

    fixed_atoms = jnp.array (json_data['fixed']['atoms']) - 1
    R0_A = jnp.array (json_data['fixed']['R0_A'])
    R0_B = jnp.array (json_data['fixed']['R0_B'])
    kval = jnp.float32 (json_data['fixed']['kval'])

    fixed_iatom = fixed_atoms[-1]
    
    ener_fun, ener_bond_fun, pairs = \
        jax_amber.get_amber_energy_funs (json_data['fname_prmtop'])

    
    def get_ener_wHO_fun (fixed_atoms, kval):
        iatom = fixed_atoms[0]
        jatom = fixed_atoms[1]
        R0 = jnp.array ([0.0, 0.0, 0.0])
        @jax.jit
        def compute_fun (R, R1):
            enr = ener_fun (R)
            enr_rst_i = ener_restraint_fun (R[iatom], R0, kval)
            enr_rst_j = ener_restraint_fun (R[jatom], R1, kval)
        
            return enr+enr_rst_i+enr_rst_j
        
        return compute_fun

    
    ener_wHO_fun = get_ener_wHO_fun (fixed_atoms, kval)
    ener_funs = (ener_wHO_fun, ener_bond_fun) 
    
    enr_bnd_A0, enr_wHO_A0 = get_energy_values (x_A, ener_funs, R0_A)
    enr_bnd_B0, enr_wHO_B0 = get_energy_values (x_B, ener_funs, R0_B)
    
    ###(TESTING)
    _, enr_wHO_A0_test = get_energy_values (tx_A, ener_funs, R0_A)
    _, enr_wHO_B0_test = get_energy_values (tx_B, ener_funs, R0_B)
    ###
    
    Z_A = x_A[:,fixed_iatom,2].mean()
    Z_B = x_B[:,fixed_iatom,2].mean()
    enr_bnd_A0 = enr_bnd_A0.mean()
    enr_bnd_B0 = enr_bnd_B0.mean()

    print (' Fixed_Z:           {:12.6f} {:12.6f}'.format(Z_A, Z_B), file=fout)
    print (' <U_wHO0>(kJ/mol):  {:12.6f} {:12.6f}'.format (enr_wHO_A0.mean(), enr_wHO_B0.mean()), file=fout)
    print ('<enr_bond>(kJ/mol): {:12.6f} {:12.6f}'.format(enr_bnd_A0, enr_bnd_B0), file=fout)

    ener_ref0 =  \
        (enr_wHO_A0, enr_wHO_B0, enr_bnd_A0, enr_bnd_B0 )
    ener_ref0_test = \
        (enr_wHO_A0_test, enr_wHO_B0_test, enr_bnd_A0, enr_bnd_B0 )
    
    lr = json_data['optax']['learning_rate']
    total_steps = json_data['optax']['total_steps']
    alpha = json_data['optax']['alpha']
    scheduler = optax.cosine_decay_schedule (lr, 
                                            decay_steps=total_steps,
                                            alpha=alpha)
    opt_method = optax.chain (
        optax.clip(0.5),
        optax.adam (learning_rate=scheduler)
    )
    #opt_method = optax.adam (learning_rate=scheduler)

    rng_seed = json_data['random_seed']
    rng = jax.random.PRNGKey(rng_seed)
    rng, key = jax.random.split (rng)
    
    input_size = x_A.shape[1]*3
    
    hidden_dim = json_data['realNVP']['hidden_dim']
    hidden_layers=json_data['realNVP']['hidden_layers']
    mask_fixed = jnp.array(json_data['realNVP']['mask_fixed']) - 1
    model = flax_nn.realNVP3(input_size=input_size, 
                     hidden_layers=hidden_layers,
                     hidden_dim=hidden_dim,
                     fixed_atoms=mask_fixed)
    
    state = train_state.TrainState.create (
            apply_fn=model.apply,
            params=model.init (key, x_A)['params'],
            tx=opt_method
        )
    
    
        
    @jax.jit
    def train_step (state, batchs,
                    ener_batch_ref0, fixed_R0):
        def loss_fn (params, apply_fn):
            b_A, b_B = batchs
        
            m_B, log_J_F = apply_fn ({'params':params}, b_A)
            m_A, log_J_R = apply_fn ({'params':params}, b_B, reverse=True)
                
            loss_wBnd, loss = loss_value (ener_wHO_fun, ener_bond_fun, ener_batch_ref0,
                    m_B, log_J_F, m_A, log_J_R, fixed_R0)
                
            return loss_wBnd

        grads = jax.grad (loss_fn) (state.params, state.apply_fn)
    
        return state.apply_gradients (grads=grads)
    
    test_ckpt = {'params': state.params, 
                'opt_state':state.opt_state}
    
    if json_data['restart_nn']['run']:
        ckpt = checkpoint_load (json_data['restart_nn']['fname_nn_pkl'])

        state = state.replace (step=state.step, 
                                params=ckpt['params'], 
                                opt_state=ckpt['opt_state'])
    
    fixed_R0 = (R0_A, R0_B)

    loss_old, _, loss_test_min, _ = \
        get_current_loss_values (state, inputs, inputs_test, 
                                 ener_wHO_fun, ener_bond_fun,
                                 ener_ref0, ener_ref0_test,
                                 fixed_R0)
    
    
    loss_test_list = []
    for epoch in range (json_data['nepoch']):
        
        for ist0 in range (0,nconf,1000):
            ied0 = ist0 + 1000
            ied0 = jnp.where (ied0 < nconf, ied0, nconf)
            batch = (x_A[ist0:ied0], x_B[ist0:ied0])
            ener_batch_ref0 = (enr_wHO_A0[ist0:ied0], 
                             enr_wHO_B0[ist0:ied0], 
                             enr_bnd_A0, enr_bnd_B0)

            state = train_step (state, batch,
                                ener_batch_ref0, fixed_R0)

        
        if (epoch+1)%10 == 0:
            
            loss_Wbnd, loss, loss_test, (m_A, m_B) = \
                get_current_loss_values (state, inputs, inputs_test, 
                                 ener_wHO_fun, ener_bond_fun,
                                 ener_ref0, ener_ref0_test,
                                 fixed_R0)
                
            diff = loss_Wbnd - loss_old 
            loss_old = loss_Wbnd

            print ('loss {:8d} {:12.4f} {:12.4f} {:12.4f} {:14.4f}'.format(
                    epoch+1, loss, loss_Wbnd, diff, loss_test),file=fout)
            
            if loss_test < loss_test_min:
                loss_test_min = loss_test 
                test_ckpt = {'params': state.params, 
                            'opt_state':state.opt_state}

                
            if loss < jnp.float32(0.0):
                break 

            loss_test_list.append (loss_test)
            
        
        if (epoch+1)%200 == 0:
            print_progress (state, inputs,
                            ener_funs,
                            ener_ref0,
                            fixed_iatom,
                            fixed_R0, 
                            fout)
            loss_test = jnp.array (loss_test_list).min()
            
            print ('loss_test {:12.4f} {:12.4f}'.format(loss_test, loss_test_min), file=fout)
            
            loss_test_list = []

        if (epoch+1)%1000 == 0:
            ckpt = {'params': state.params, 'opt_state': state.opt_state}
            checkpoint_save (json_data['fname_nn_pkl'], ckpt)
            checkpoint_save (json_data['fname_nn_test_pkl'], test_ckpt)
            write_traj (json_data['fname_mA_dcd'], m_A)
            write_traj (json_data['fname_mB_dcd'], m_B)
            
        
    ckpt = {'params': state.params, 'opt_state': state.opt_state}
    checkpoint_save (json_data['fname_nn_pkl'], ckpt)
    checkpoint_save (json_data['fname_nn_test_pkl'], test_ckpt)

    state = state.replace (step=0,
                            params=test_ckpt['params'],
                            opt_state=test_ckpt['opt_state'])
    print ("===SUMMARY===", file=fout)
    print_progress (state, inputs,
                    ener_funs,
                    ener_ref0,
                    fixed_iatom,
                    fixed_R0, 
                    fout)


if __name__ == '__main__':
    import json 
    import sys

    fname_json = 'input.json'
    if len(sys.argv) > 1:
        fname_json = sys.argv[1]
    
    with open(fname_json) as f:
        json_data = json.load (f)
        main_train (json_data)
