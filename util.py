import mdtraj as md
from mdtraj.formats import DCDTrajectoryFile
import jax.numpy as jnp
import pickle

def get_trajectory (fname_prmtop, fname_dcd, nsamp):

    c = md.load (fname_dcd, top=fname_prmtop)
    crds = jnp.array (c.xyz)
    #TODO: Modify the samped data
    #return crds[-nsamp:], crds[:-nsamp] # in nm unit
    return crds[-nsamp:], crds[:1000]


def write_traj (fname, traj_xyz_nm):
    traj_xyz = traj_xyz_nm*10
    n_conf = traj_xyz.shape[0]
    if traj_xyz.shape[-1] != 3:
        traj_xyz = traj_xyz.reshape(n_conf, -1, 3)
    
    with DCDTrajectoryFile(fname, 'w') as f:
        f.write (traj_xyz)


def checkpoint_save (fname, ckpt):
    with open(fname, 'wb') as fp:        
        pickle.dump (ckpt, fp)


def checkpoint_load (fname):
    with open (fname, 'rb') as fp:
        return pickle.load (fp)
    