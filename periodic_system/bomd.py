#from mpi4py import MPI
#from pyscf.data.nist import BOLTZMANN, HARTREE2J, HBAR
"""
Length Unit: nm
Energy Unit: kJ/mol
Force Unit : kJ/(nm mol): dE/dx
Time Unit : ps
Velocity Unit: nm/ps

mass unit : amu = 1 g/mol
Note: 1 kJ = 1 kg [m/s]^2 = g [nm/ps]^2
Force Unit/Mass Unit: [kJ/(nm mol)] / [mol/g] = nm/ps^2
V (nm/ps) = F/M dt

"""
import jax
import jax.numpy as jnp
import numpy as np 
import dataclasses
import sys

NNHC = 4
kB = 8.314462618e-3 # kJ/(mol K)
kBT = kB*jnp.float32(300.0)


def kinetic_energy(mass, vel):
    """ calculate a kinetic energy
        Parameters
        -----
        vel : (natom, 3) velocity
        mass : (natom)   masses 

        Return
        -----
        kinetic energy : float
    """
        
    return 0.5*jnp.einsum('i,ij,ij', mass, vel, vel)


def vel_init(mass):
    """ Initiate the velocities
    Parameters
        -----
    mass_list : (natom) : the mass of the atom (amu)

    Return
    -----
    vel : (natom, 3) velocities
    """

    natom = mass.shape[0]
    mass_inv = 1.0/mass
    vsigma = jnp.sqrt(kBT*mass_inv)
    # generate initial velocities with mu = 0 and sigma = 0.3
    
    rng = jax.random.PRNGKey(2)
    rng_vel, rng = jax.random.split(rng, num=2)
    vel_rnd = jax.random.normal(rng_vel,(natom, 3) )*jnp.float32(0.3)

    vel  = jnp.einsum('i,ij->ij', vsigma, vel_rnd)
    sump = jnp.einsum('i,ij->j', mass, vel)/natom
    vel= vel - jnp.einsum('j,i->ij', sump, mass_inv)

    ekin = kinetic_energy(mass, vel)
    ekin = 2.0*ekin / (3*natom)
    scale = jnp.sqrt(kBT/ekin)

    return scale*vel

@dataclasses.dataclass
class NHC:
    q_pos: np.ndarray
    q_vel: np.ndarray 
    q_mass: np.ndarray 
    ekin: np.float32
    dof: np.int32 
    

def nhc_init(ekin, gfree):
    """ Initiate the thermal bath of Nose-Hoover Chain Method
    Parameters
    -----
    ekin : float : The kinetic energy of the system
    gfree : int  : The degree of freedom of the system (3*natom-5)
    
    Return
    -----
    qmass: (NNHC) : the mass for the thermal bath.
    """

    time_system = 8.881e-3 # ps 
    omega_system = 2.0*np.pi/time_system
    omega2 = omega_system*omega_system
    ekin_omega2 = kBT / omega2
    
    q_mass = np.array ([gfree, 1.0, 1.0, 1.0])*ekin_omega2
    #q_mass = np.ones(NNHC, dtype=np.float32) * ekin_omega2
    #q_mass[0] = ekin_omega2*gfree
    q_pos = np.zeros (NNHC, dtype=np.float32)
    q_vel = np.zeros (NNHC, dtype=np.float32)
    
    return NHC (q_pos, q_vel, q_mass, ekin, gfree)
    

def nhc_half_step (dt, vel, state):

    q_pos, q_vel, q_mass, ekin, gfree = dataclasses.astuple(state)
    
    dt2 = 0.5*dt
    dt4 = 0.5*dt2
    dt8 = 0.5*dt4

    #ekin = kinetic_energy(mass, vel)

    q_frc = np.empty(NNHC, dtype=np.float32)
    
    q_frc[0] =(2.0*ekin - gfree*kBT)/q_mass[0]
    for ihc in range (1, NNHC):
        q_frc[ihc] = (q_mass[ihc-1]*q_vel[ihc-1]**2 - kBT)/q_mass[ihc]

    q_vel[-1] = q_vel[-1] + q_frc[-1]*dt4
        
    for jhc in range(NNHC-1, 0,-1):
        # jhc = [3,2,1]
        scale = np.exp(-q_vel[jhc]*dt8)
        vtmp = q_vel[jhc-1]
        q_vel[jhc-1] = scale*(scale*vtmp + q_frc[jhc-1]*dt4)

    # update atomic velocities
    scale = np.exp(-q_vel[0]*dt2)

    ekin = ekin*scale**2
    vel = vel*scale
    
    q_pos = q_pos + dt2*q_vel 
    
    q_frc[0] = (2.0*ekin - gfree*kBT)/q_mass[0]

    for ihc in range(NNHC-1):
        scale = np.exp(-q_vel[ihc+1]*dt8)
        vtmp = q_vel[ihc]
        q_vel[ihc] = scale*(scale*vtmp + q_frc[ihc]*dt4)
        q_frc[ihc+1] = (q_mass[ihc]*q_vel[ihc]**2 - kBT)/q_mass[ihc+1]

    q_vel[-1] = q_vel[-1] + q_frc[-1]*dt4

    return vel, NHC (q_pos, q_vel, q_mass, ekin, gfree)


def nhc_tot_energy (state):
    q_pos, q_vel, q_mass, ekin, gfree = dataclasses.astuple(state)
    
    q_kin = 0.5*np.sum(q_mass*q_vel**2, axis=-1)
    dof = np.array ([gfree, 1., 1., 1.])
    q_pot = kBT*np.sum(dof*q_pos, axis=-1)

    return q_kin, q_pot


def bomd (mass, ener_grad_fn, box):

    m_mass_inv = 1.0/mass
    m_mass = mass 
    # K in kJ/mol
    m_gfree = 3.0*mass.shape[0]

    m_dt = 0.5e-3 # ps
    m_dt_2 = 0.5*m_dt
    
    
    def run_verlet_velocity(crds, md_nstep, nstep_print=10):

        fout_dat = open('md_nvt_vv.dat', 'w', 1)
        print ("    NSTEP      EPOT          EKIN          TEMP        ETOT          HTOT ", file=fout_dat)
        vel = vel_init(m_mass)
        ekin = kinetic_energy (m_mass, vel)
        nhc_state = nhc_init (ekin, m_gfree)

        enr, grd = ener_grad_fn (crds, box)
        acc = -np.einsum('ij,i->ij',grd, m_mass_inv)

        for istep in range(md_nstep):

            nhc_state.ekin = kinetic_energy (m_mass, vel)
            vel, nhc_state = nhc_half_step (m_dt, vel, nhc_state)
            
            crds = crds + m_dt*(vel + acc * m_dt_2)
            acc_ave = 0.5*acc 

            enr, grd = ener_grad_fn (crds, box)
            acc = -np.einsum('ij,i->ij',grd, m_mass_inv)
            acc_ave += 0.5*acc 
            vel += acc_ave * m_dt

            nhc_state.ekin = kinetic_energy (m_mass, vel)
            vel, nhc_state = nhc_half_step (m_dt, vel, nhc_state)
            
            if (istep+1)%nstep_print == 0:
                
                temp = 2.0*nhc_state.ekin /(m_gfree*kB)
                h_sys = enr + nhc_state.ekin 
                #---
                q_kin, q_pot = nhc_tot_energy(nhc_state)
                h_tot = h_sys + q_kin + q_pot
                print ("{:10d} {:12.6f} {:12.6f} {:10.4f} {:12.6f} {:12.6f}".format(istep+1, 
                                                                   enr, 
                                                                   nhc_state.ekin, 
                                                                   temp, 
                                                                   h_sys, h_tot), file=fout_dat)
 
    return run_verlet_velocity


if __name__ == '__main__':
    import mdtraj as md
    import jax_amber
    
    fname_prmtop = 'ala_deca_peptide_wat.prmtop'
    fname_pdb = 'ala_deca_peptide_wat.pdb'
    
    c = md.load (fname_pdb, top=fname_prmtop)
    crds = jnp.array (c.xyz[0]) # lenght unit is nm

    ener_fun, masses = jax_amber.get_amber_energy_fun (fname_prmtop, l_pbc=True)
    ener_grad_fun = jax.value_and_grad (ener_fun)
    box = jnp.array([3.0, 3.0, 4.5]) #  (90.0, 90.0, 90.0) A --> (9.0, 9.0, 9.0) nm 
    md_vv_nvt = bomd (masses, ener_grad_fun, box)
    md_nstep = 50
    
    md_vv_nvt (crds, md_nstep, nstep_print=2)
