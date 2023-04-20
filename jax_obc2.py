import jax.numpy as jnp 
import jax
#import numpy as np 

"""

 OBC METHOD
 A. Onufriev, D. Bashford and D. A. Case, "Exploring Protein
     Native States and Large-Scale Conformational Changes with a
     Modified Generalized Born Model", PROTEINS, 55, 383-394 (2004)

 ACE METHOD
 M. Schaefer, C. Bartels and M. Karplus, "Solution Conformations
     and Thermodynamics of Structured Peptides: Molecular Dynamics
     Simulation with an Implicit Solvation Model", J. Mol. Biol.,
     284, 835-848 (1998) 

"""
E_CHARGE = 1.602176634e-19
AVOGADRO = 6.02214076e23
# (e^2 Na/(kJ nm)) == (e^2/(kJ mol nm)) */ 
EPSILON0 = 1e-6*8.8541878128e-12/(E_CHARGE*E_CHARGE*AVOGADRO)
ONE_4PI_EPS0 = 1.0/(4.0*jnp.pi*EPSILON0)


def prm_get_gb_parms (prm_raw_data):
    
    screen = [float(s) for s in prm_raw_data['SCREEN']]
    radii  = [float(r)/10 for r in prm_raw_data['RADII']]
    
    return (jnp.array(radii), jnp.array(screen))


def ener_gbsa_obc2 (chgs, gb_parms):
    # The following values are from 'ObcParameters.cpp'

    radii, scaled_factor = gb_parms 
    # OpenMM used water dielectric constant : 78.5
    # ddcosmo used : 78.3553
    solvent_dielectric = jnp.float32(78.5) # or 78.3553
    solute_dielectric = jnp.float32(1.0)
    electric_constant = -jnp.float32(0.5)*ONE_4PI_EPS0
    preFactor = 2.0*electric_constant* \
            (1.0/solute_dielectric - 1.0/solvent_dielectric)
    probe_radius = jnp.float32(0.14)
    pi4_Asolv = jnp.float32(28.3919551)
    dielectric_offset = jnp.float32(0.009) # nm
    rad_offset = radii - dielectric_offset # (nAtoms)
    rad_offset_i = rad_offset.reshape(-1,1) # (nAtoms, 1)
    rad_scaled = jnp.einsum('i,i->i', rad_offset, scaled_factor)
    rad_scaled2 = jnp.einsum('i,i->i', rad_scaled, rad_scaled)

    #nAtoms = radii.shape[0]

    # OBC2 Parameters
    alphaObc = jnp.float32(1.0)
    betaObc = jnp.float32(0.8)
    gammaObc = jnp.float32(4.85)

    def get_born_radii (R):
        # rad_offset_i : radius_offset at atom i (Float)
        # rij : the inter-distance between atom i and others : Array (Float)
        '''
        rij = []
        for ia in range (nAtoms):
            rab = R[ia] - R
            dab = jax_md.space.distance (rab)
            rij.append(dab)
        rij = jnp.array(rij)
        '''
        rij = jnp.linalg.norm(R.reshape(-1,1,3) - R, axis=2)

        # shape = (n_atom,n_atom)
        rij_rscaled = rad_scaled + rij
        diff_rad = rad_scaled - rij
        abs_diff_rad = abs (diff_rad)
        
        l_ij = 1.0/jnp.maximum (rad_offset_i, abs_diff_rad)
        u_ij = 1.0/rij_rscaled 
        l_ij2 = l_ij*l_ij
        u_ij2 = u_ij*u_ij 

        inv_rij = 1.0/rij 
        log_ratio = jnp.log ( (u_ij/l_ij) )
        term = l_ij - u_ij + 0.25*rij*(u_ij2-l_ij2) + \
                0.5*inv_rij*log_ratio + (0.25*rad_scaled2*inv_rij)*(l_ij2-u_ij2)
        term2 = 2.0*(1.0/rad_offset_i - l_ij)

        zeros = jnp.float32(0.0) #jnp.zeros (rij.shape[0])

        #  shape : (n_atom, n_atom) --> (n_atom)
        sum = jnp.sum (jnp.where (rad_offset_i<rij_rscaled, term, zeros), axis=-1)
        sum += jnp.sum (jnp.where (rad_offset_i<diff_rad, term2, zeros), axis=-1)

        sum *= 0.5*rad_offset
        sum2 = sum*sum
        sum3 = sum2*sum
        tanh_sum = jnp.tanh(alphaObc*sum - betaObc*sum2 + gammaObc*sum3)
        
        born_radii = 1.0/(1.0/rad_offset - tanh_sum/radii) 
        
        return born_radii


    def get_ACE_nonpolar_energy (ri, rb):
        return pi4_Asolv * (ri+probe_radius)**2 * (ri/rb)**6
        

    def compute_fn (R, born_radii):
        
        #print ('start ACE')
        ener_ace = jax.vmap (get_ACE_nonpolar_energy) (radii, born_radii)
        ener_ace = jnp.sum(ener_ace)

        #print ('start POLE')
        # GB electrostatic polarization energy term 
        def get_GB_polar_energy (chg_i, r_i, born_rad_i):
            pchg_ij = preFactor*chg_i*chgs
            drab = r_i - R
            r_ij2 = jnp.sum(drab**2, axis=-1)
            alpha2 = born_rad_i*born_radii 
            expTerm = jnp.exp(-0.25*r_ij2/alpha2)
            fgb = jnp.sqrt (r_ij2 + alpha2*expTerm)

            return jnp.float32(0.5)*jnp.sum(pchg_ij/fgb)

        ener_obc2 = jax.vmap(get_GB_polar_energy) (chgs, R, born_radii)
        ener_obc2 = jnp.sum(ener_obc2)

        return ener_ace + ener_obc2

    return compute_fn, get_born_radii


if __name__ == '__main__':
    import mdtraj as md
    import jax_amber 

    fname_dcd = 'traj_complex.dcd'
    fname_prmtop = 'da.prmtop'

    c = md.load (fname_dcd, top=fname_prmtop)
    crds = jnp.array (c.xyz) # lenght unit is nm

    prm_raw_data = jax_amber.amber_prmtop_load(fname_prmtop)
    chgs = jax_amber.prm_get_charges (prm_raw_data)
    gb_parms = prm_get_gb_parms (prm_raw_data)

    ener_gbsa_obc2_fn, get_born_radii = ener_gbsa_obc2 (chgs, gb_parms)
    ener_gbsa_obc2_fn = jax.jit (ener_gbsa_obc2_fn)

    born_radii = jax.vmap (get_born_radii) (crds)
    en_obc2 = jax.vmap(ener_gbsa_obc2_fn) (crds, born_radii)
    
    print ('<en_obc2>(kJ/mol)', en_obc2.mean()/4.184)
