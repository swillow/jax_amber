import jax
import jax.numpy as jnp
import sys

_ONE_4PI_EPS0 = jnp.float32(138.935456)

def get_ewald_fun (box_init,
                   chgs,
                   eps_ewald=jnp.float32(1.0e-6), 
                   r_cut=jnp.float32(1.0)):

    #box = jnp.array ([100., 80., 60.])
    box = box_init
    boxi = 1.0/box_init
    
    twopi = jnp.float32(2.0)*jnp.pi
    ###
    pp = -jnp.log (eps_ewald)
    aewald = jnp.sqrt (pp)/r_cut # length unit is A
    kvec_cut = 2.0*aewald*jnp.sqrt(pp)
    kvec_cutSQ = kvec_cut**2
    fac_ew2 = -0.25/(aewald*aewald)
    print ('PP', pp)
    print ('kvec_cut', kvec_cut)
    hmax = int (kvec_cut/twopi*box[0])+1
    kmax = int (kvec_cut/twopi*box[1])+1
    lmax = int (kvec_cut/twopi*box[2])+1
    print ('hkl', hmax, kmax, lmax)
    kmax2 = 2*kmax + 1
    lmax2 = 2*lmax + 1
    
    factor = 2.0
    k_hkl = jnp.zeros ( (hmax*kmax2*lmax2,3), dtype=jnp.int64)

    k_hkl = k_hkl.at[:,2].set ( jnp.array([[i] for i in range(hmax*kmax2*lmax2)]).reshape(-1)%lmax2 - lmax)
    k_hkl = k_hkl.at[:,1].set ( jnp.array([lmax2*[i] for i in range(hmax*kmax2)]).reshape(-1)%kmax2 - kmax)
    k_hkl = k_hkl.at[:,0].set ( jnp.array([kmax2*lmax2*[i] for i in range(hmax)]).reshape(-1))
    k_hkl = k_hkl[kmax*lmax2+lmax+1:]
    kv_hkl = twopi*boxi*k_hkl

    kv2_hkl = jnp.einsum('ij,ij->i', kv_hkl, kv_hkl)
    select = kv2_hkl < kvec_cutSQ
    print ('out', select.sum())
    k_hkl = k_hkl[select]

    print (k_hkl.shape)
    print (k_hkl[:3])
    #kv_hkl = kv_hkl[select]
    
    #chgs = jnp.array ([1,2,3])

    def eir_setup (R, box0=None):
        if box0 is None:
            boxi = 1.0/box
        else:
            boxi = 1.0/box0 

        nAtoms = R.shape[0]
        # (nAtoms, 3)
        ang_val = twopi*jnp.einsum('j,ij->ij',boxi,R)
        eir_x = jnp.ones ( (hmax, nAtoms), dtype=jnp.cdouble)
        eir_y = jnp.ones ( (kmax, nAtoms), dtype=jnp.cdouble)
        eir_z = jnp.ones ( (lmax, nAtoms), dtype=jnp.cdouble)
    
        eir1_x = jax.lax.complex (jnp.cos(ang_val[:,0]), jnp.sin(ang_val[:,0]))
        eir1_y = jax.lax.complex (jnp.cos(ang_val[:,1]), jnp.sin(ang_val[:,1]))
        eir1_z = jax.lax.complex (jnp.cos(ang_val[:,2]), jnp.sin(ang_val[:,2]))

        eir_x = eir_x.at[1].set (eir1_x)
        eir_y = eir_y.at[1].set (eir1_y)
        eir_z = eir_z.at[1].set (eir1_z)
    
        for kx in range (2, hmax):
            eir_x = eir_x.at[kx].set (eir_x[kx-1]*eir1_x)
        for ky in range (2, kmax):
            eir_y = eir_y.at[ky].set (eir_y[ky-1]*eir1_y)
        for kz in range (2, lmax):
            eir_z = eir_z.at[kz].set (eir_z[kz-1]*eir1_z)
    
        return eir_x, eir_y, eir_z

    def compute_fun (R, box0=None):

        if box0 is None:
            boxi = 1.0/box 
            Vol = box[0]*box[1]*box[2]
        else:
            boxi = 1.0/box0
            Vol = box0[0]*box0[1]*box0[2]

        eir_x, eir_y, eir_z = eir_setup(R)
        
        @jax.jit 
        def ewald_std_qq_kvec (k_hkl0):
            kx, ky, kz = k_hkl0
            
            kvec0 = twopi*boxi*k_hkl0
            kv2 = jnp.einsum('i,i->', kvec0, kvec0)
            Ak0 = factor*jnp.exp(fac_ew2*kv2)/kv2 
            
            p4vAkk = 2.0*twopi*Ak0/Vol
            
            tmp_ky = jnp.where (ky<0, eir_y[-ky].conjugate(), eir_y[ky])
            tmp_kz = jnp.where (kz<0, eir_z[-kz].conjugate(), eir_z[kz])
            
            tab_kxky = eir_x[kx]*tmp_ky
            tab_kr = tab_kxky*   tmp_kz 
            
            rQqk = jnp.sum(chgs*jnp.real(tab_kr))
            iQqk = jnp.sum(chgs*jnp.imag(tab_kr))
            
            tmp = p4vAkk*(jnp.real(tab_kr)*iQqk - jnp.imag(tab_kr)*rQqk)
            Efq = jnp.einsum('i,j->ij',tmp,kvec0)
            
            phi = p4vAkk*(jnp.real(tab_kr)*rQqk + jnp.imag(tab_kr)*iQqk)
            #phi = phi.at[:nWat].set(0.0)
            return Efq, phi

        # Self Energy 
        U_self = - _ONE_4PI_EPS0*aewald/jnp.sqrt(jnp.pi)*jnp.einsum('i,i->', chgs, chgs)
        # Reciprocal Space
        # Efq(kpoints, nAtoms, 3), phi (kpoints, nAtoms)
        Efq, phi = jax.vmap(ewald_std_qq_kvec) (k_hkl)

        Efq = Efq.sum(axis=0)
        phi = phi.sum(axis=0)
        
        Uelec = jnp.float32(0.5)*_ONE_4PI_EPS0*jnp.einsum('i,i->', chgs, phi)
        grad = -_ONE_4PI_EPS0*jnp.einsum('i,ij->ij',chgs,Efq)
        return Uelec+U_self, grad 
    
    return compute_fun 



if __name__ == '__main__':
    rcut = 1.0 # nm
    eps_ewald = 1.e-6 # unit??

    fname = 'test/water.xyzq'
    lines = open(fname, 'r').readlines()
    natoms = int (lines[0].split()[0])
    print (natoms)
    words = lines[1].split()
    box = 0.1*jnp.array ([float(words[0]), float(words[1]), float(words[2])])

    crds = []
    chgs = []
    for line in lines[2:]:
        words = line.split()
        x = float (words[0])
        y = float (words[1])
        z = float (words[2])
        q = float (words[3])
        crds.append ([x, y, z])
        chgs.append (q)
    
    crds = 0.1*jnp.array (crds)
    chgs = jnp.array(chgs)
    print ('crds', crds.shape)
    ener_ewald_fn = get_ewald_fun (box, chgs, eps_ewald, r_cut=rcut)

    enr, grad = ener_ewald_fn (crds, box)
    print (enr, 'kJ/mol', enr/4.184, 'kcal/mol')
