import numpy as np
import jax.numpy as jnp
import jax
import re
import sys

_ONE_4PI_EPS0 = jnp.float32(138.935456)

def amber_prmtop_load (fname_prmtop):
    ''' From openmm/wrappers/python/openmm/app/internal/amber_file_parser.py
    fname_prmtop: string 
    '''
    FORMAT_RE_PATTERN=re.compile("([0-9]+)\(?([a-zA-Z]+)([0-9]+)\.?([0-9]*)\)?")
    
    _flags = []
    _raw_data = {}
    _raw_format = {}
    with open (fname_prmtop, 'r') as fIn:
        for line in fIn:
            if line[0] == '%':
                if line.startswith('%VERSION'):
                    tag, _prmtopVersion = line.rstrip().split(None, 1)
                elif line.startswith('%FLAG'):
                    tag, flag = line.rstrip().split(None, 1)
                    _flags.append (flag)
                    _raw_data[flag] = []
                elif line.startswith('%FORMAT'):
                    format = line.rstrip()
                    index0 = format.index('(')
                    index1 = format.index(')')
                    format = format[index0+1:index1]
                    m = FORMAT_RE_PATTERN.search(format)
                    _raw_format[_flags[-1]] = (format, m.group(1), m.group(2),
                                            int(m.group(3)), m.group(4) )
                elif line.startswith('%COMMENT'):
                    continue
            elif _flags \
                and 'TITLE' == _flags[-1] \
                    and not _raw_data['TITLE']:
                    _raw_data['TITLE'] = line.rstrip()
            else:
                flag = _flags[-1]
                (format, numItems, itemType, iLength, itemPrecision) = _raw_format[flag]
                line = line.rstrip()
                for index in range (0, len(line), iLength):
                    item = line[index:index+iLength]
                    if item:
                        _raw_data[flag].append(item.strip())
                
    return _raw_data



def prm_get_charges (prm_raw_data):
    charge_list = [float(x)/18.2223 for x in prm_raw_data['CHARGE']]
    return jnp.array(charge_list)


def prm_get_atom_types (prm_raw_data):
    atomType = [ int(x) - 1 for x in prm_raw_data['ATOM_TYPE_INDEX']]
    return jnp.array (atomType)


def prm_get_nonbond_terms (prm_raw_data):
    numTypes = int(prm_raw_data['POINTERS'][1])

    LJ_ACOEF = prm_raw_data['LENNARD_JONES_ACOEF']
    LJ_BCOEF = prm_raw_data['LENNARD_JONES_BCOEF']
    # kcal/mol --> kJ/mol
    energyConversionFactor = 4.184
    # A -> nm
    lengthConversionFactor = 0.1

    sigma = np.zeros (numTypes)
    epsilon = np.zeros (numTypes)

    for i in range(numTypes):
        
        index = int (prm_raw_data['NONBONDED_PARM_INDEX'][numTypes*i+i]) - 1    
        acoef = float(LJ_ACOEF[index])
        bcoef = float(LJ_BCOEF[index])

        try:
            sig = (acoef/bcoef)**(1.0/6.0)
            eps = 0.25*bcoef*bcoef/acoef
        except ZeroDivisionError:
            sig = 1.0
            eps = 0.0

        sigma[i] = sig*lengthConversionFactor
        epsilon[i] = eps*energyConversionFactor

    return jnp.array(sigma), jnp.array(epsilon)



def prm_get_nonbond_pairs (prm_raw_data):
    num_excl_atoms = prm_raw_data['NUMBER_EXCLUDED_ATOMS']
    excl_atoms_list = prm_raw_data['EXCLUDED_ATOMS_LIST']
    total = 0
    numAtoms = int(prm_raw_data['POINTERS'][0])
    nonbond_pairs = []
        
    for iatom in range(numAtoms):
        index0 = total
        n = int (num_excl_atoms[iatom])
        total += n
        index1 = total
        excl_list = []
        for jatom in excl_atoms_list[index0:index1]:
            j = int(jatom) - 1
            excl_list.append(j)

        for jatom in range (iatom+1, numAtoms):
            if jatom in excl_list:
                continue
            nonbond_pairs.append ( [iatom, jatom] )
        
    return jnp.array(nonbond_pairs)




def prm_get_nonbond14_info (prm_raw_data):
    
    dihedralPointers = prm_raw_data["DIHEDRALS_WITHOUT_HYDROGEN"] + \
                            prm_raw_data["DIHEDRALS_INC_HYDROGEN"] 

    nonbond14_pairs = []
    

    for ii in range (0, len(dihedralPointers), 5):
        if int(dihedralPointers[ii+2])>0 and int(dihedralPointers[ii+3])>0:
            iAtom = int(dihedralPointers[ii])//3
            lAtom = int(dihedralPointers[ii+3])//3
            
            nonbond14_pairs.append ( (iAtom, lAtom) )

    return jnp.array(nonbond14_pairs)


def square_distance (dR:jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(dR**2, axis=-1)


def distance (dR:jnp.ndarray):
    dr = square_distance(dR)
    return jnp.sqrt(dr)

def periodic_distance (Box:jnp.ndarray, dR:jnp.ndarray):
    """
    Args:
    Box: Box length (ndarray(3))
    dR : distance between two particles (ndarray(shape=[..., 3]))
    Returns:
    The shortest distance between two particles
    """
    return jnp.mod (dR+Box*jnp.float32(0.5), Box) - jnp.float32(0.5)*Box 


def cosine_angle_between_two_vectors(dR_12: jnp.ndarray, dR_13: jnp.ndarray) -> jnp.ndarray:
    dr_12 = distance(dR_12) + 1e-7
    dr_13 = distance(dR_13) + 1e-7
    cos_angle = jnp.dot(dR_12, dR_13) / dr_12 / dr_13
    return jnp.clip(cos_angle, -1.0, 1.0)


def harmonic_interaction (r, r0, k0):
    # U = 0.5 * k0 * (r - r0)^2
    return jnp.float32(0.5)*k0*(r - r0)**2


def torsion_interaction (theta, cos_phase0, n0, k0):
    # theta : torsional angle
    # cos_phase0 = cos (theta0), where theta0 is 0 or pi
    # U_torsion = k0 * (1 + cos (n0 theta - theta0))
    # U_torsion = k0 * (1 + cos (n0 theta) * cos_phase0)

    return k0*(1.0 + jnp.cos(n0*theta)*cos_phase0)


def ener_bond (bonds, bond_types, rb0, kb0):
    '''
    bonds : jnp.array ( (n_bond, 2), dtype=int )
    bond_types: jnp.array ( (n_bond), dtype=int )
    rb0 : jnp.array ( (n_bond_type), dtype=float )
    kb0 : jnp.array ( (n_bond_type), dtype=float )
    '''
    r0 = rb0[bond_types]
    k0 = kb0[bond_types]

    def compute_fn (R):
        '''
        R ((n_atom, 3),dtype=float)
        '''
        dR12 = R[bonds[:,1]] - R[bonds[:,0]] # (n_bond, 3)
        
        # r (n_bond)
        r  = jax.vmap(distance) (dR12)
        # en_val (n_bond)
        en_val = jax.vmap(harmonic_interaction) (r, r0, k0)

        return jnp.sum (en_val)

    return compute_fn


def ener_angle (angles, angle_types, r_theta0, k_theta0):
    '''
    angles: jnp.array ((n_angle, 3), dtype=int) 
    angle_types: jnp.array ( (n_angle), dtype=int )
    r_theta0 : jnp.array ( (n_angle_type), dtype=float )
    k_theta0 : jnp.array ( (n_angle_torsion), dtype=float )
    '''
    theta0 = r_theta0[angle_types]
    k0 = k_theta0[angle_types]

    def compute_fn (R):
        '''
        R: jnp.array ( (n_atom, 3), dtype=float)
        '''
        dR21 = R[angles[:, 0]] - R[angles[:, 1]]# (n_angle, 3)
        dR23 = R[angles[:, 2]] - R[angles[:, 1]]

        # theta (n_angle)
        cos_angle = jax.vmap(cosine_angle_between_two_vectors) (dR21, dR23)
        theta = jnp.arccos(cos_angle)
        # en_val (n_angle)
        en_val = jax.vmap(harmonic_interaction) (theta, theta0, k0)

        return jnp.sum (en_val)

    return compute_fn



def ener_torsion (torsions, torsion_types, torsion_values):
    '''
    torsions: jnp.array ((n_torsion, 4), dtype=int) 
    torsion_types: jnp.array ( (n_torsion), dtype=int )
    n_theta0 : jnp.array ( (n_torsion_type), dtype=int )
    cos_phase0 : jnp.array ( (n_torsion_type), dtype=float )
    k_theta0 : jnp.array ( (n_torsion_torsion), dtype=float )
    '''
    n_theta0, cosine_phase0, k_theta0 = torsion_values
    n0 =  n_theta0[torsion_types]
    cos_phase0 =  cosine_phase0[torsion_types]
    k0 =  k_theta0[torsion_types]


    def theta_fn (dR12, dR23, dR34):
        '''
        Estimate torsional angle using four atoms: R1, R2, R3, R4
        R1, R2, R3, R4: jnp.array ( (3), dtype=float)
        '''
        
        dRT = jnp.cross (dR12, dR23)
        dRU = jnp.cross (dR23, dR34)
        dTU = jnp.cross (dRT, dRU)

        rt = distance (dRT) + 1.e-7
        ru = distance (dRU) + 1.e-7
        r23 = distance (dR23) + 1.e-7

        cos_angle = jnp.dot (dRT, dRU)/(rt*ru)
        sin_angle = jnp.dot (dR23, dTU)/(r23*rt*ru)
        theta = jnp.arctan2(sin_angle, cos_angle)

        return theta


    def compute_fn (R):
        '''
        R: jnp.array( (n_atom, 3), dtype=float)
        '''
        dR12 = R[torsions[:,1]] - R[torsions[:,0]] # R1 (n_torsion, 3)
        dR23 = R[torsions[:,2]] - R[torsions[:,1]]
        dR34 = R[torsions[:,3]] - R[torsions[:,2]]
        
        # theta (n_torsion)
        theta = jax.vmap (theta_fn) (dR12, dR23, dR34)
        # en_val (n_torsion)
        en_val = jax.vmap(torsion_interaction) (theta, cos_phase0, n0, k0)

        return jnp.sum(en_val)
    
    return compute_fn



def get_bonds_info (prm_raw_data):
        
    # kcal/mol/A^2 --> kJ/mol/nm^2
    forceConstConversionFactor = jnp.float32 (418.4)
    # Amber : k(r - r0)^2
    # openmm and this code : 0.5 * k' (r - r0)^2
    # k' = 2 * k    
    forceConstant = jnp.float32(2.0)*jnp.array(
            [float(k0) for k0 in prm_raw_data['BOND_FORCE_CONSTANT']] 
        )*forceConstConversionFactor
        
    # A --> nm
    lengthConversionFactor = jnp.float32 (0.1)
    bondEquil = jnp.array(
            [float(r0) for r0 in prm_raw_data['BOND_EQUIL_VALUE']]
        )*lengthConversionFactor       
        
    bondPointers = prm_raw_data['BONDS_WITHOUT_HYDROGEN'] + \
        prm_raw_data['BONDS_INC_HYDROGEN']
        
    bonds = []
    bond_types = []
    for ii in range (0, len(bondPointers), 3):
        iType = int (bondPointers[ii+2]) - 1
        bonds.append ( (int(bondPointers[ii])//3,
                        int(bondPointers[ii+1])//3) )
        bond_types.append(iType)

    return jnp.array(bonds), \
           jnp.array(bond_types), \
           bondEquil, forceConstant



def get_angles_info (prm_raw_data):

    # kcal/mol/rad^2 --> kJ/mol/rad^2
    forceConstConversionFactor = jnp.float32 (4.184) 
    # Amber : k(r - r0)^2
    # openmm and this code : 0.5 * k' (r - r0)^2
    # k' = 2 * k
    forceConstant = jnp.float32(2.0)*jnp.array(
            [float(k0) for k0 in prm_raw_data['ANGLE_FORCE_CONSTANT']] 
        )*forceConstConversionFactor
        
    angleEquil = jnp.array(
            [float(r0) for r0 in prm_raw_data['ANGLE_EQUIL_VALUE']]
        )
        

    anglePointers = prm_raw_data['ANGLES_WITHOUT_HYDROGEN'] + \
        prm_raw_data['ANGLES_INC_HYDROGEN']
        
    angles = []
    angle_types = []
    for ii in range (0, len(anglePointers), 4):
        iType = int (anglePointers[ii+3]) - 1
        angles.append ( (int(anglePointers[ii]) //3,
                        int(anglePointers[ii+1])//3,
                        int(anglePointers[ii+2])//3) )
        angle_types.append(iType)

    return jnp.array(angles), \
            jnp.array(angle_types),\
            angleEquil, forceConstant



def get_dihedrals_info (prm_raw_data):
    
    # kcal/mol/rad^2 --> kJ/mol/rad^2
    forceConstConversionFactor = jnp.float32 (4.184) 
    forceConstant = jnp.array(
            [float(k0) for k0 in prm_raw_data['DIHEDRAL_FORCE_CONSTANT']] 
        )*forceConstConversionFactor
        
    cos0 = jnp.array([jnp.cos(float(ph0)) for ph0 in prm_raw_data['DIHEDRAL_PHASE']])
    cos_phase0 = jnp.where (cos0 < 0, jnp.float32(-1), jnp.float32(1.0))

    periodicity = jnp.array(
            [int (0.5 + float(n0)) for n0 in prm_raw_data['DIHEDRAL_PERIODICITY']]
        )
        
    dihedralPointers = prm_raw_data['DIHEDRALS_WITHOUT_HYDROGEN'] + \
        prm_raw_data['DIHEDRALS_INC_HYDROGEN']
        
    dihedrals = []
    dihedral_types = []
    for ii in range (0, len(dihedralPointers), 5):
        iType = int (dihedralPointers[ii+4]) - 1
        dihedrals.append ( (int(dihedralPointers[ii]) //3,
                                int(dihedralPointers[ii+1])//3,
                            abs(int(dihedralPointers[ii+2]))//3,
                            abs(int(dihedralPointers[ii+3]))//3) )
        dihedral_types.append(iType)

    dihedral_type_values = (periodicity, cos_phase0, forceConstant)
        
    return jnp.array(dihedrals), \
           jnp.array(dihedral_types), \
           dihedral_type_values



def ener_bonded (prm_raw_data):
    '''
    prmtop._raw_data: dict{} : from amber prmtop file
    ener_bonded = ener_bond + ener_angle + ener_torsion
    '''
    bonds, bond_types, r_bond0, k_bond0 = get_bonds_info(prm_raw_data)
    ener_bond_fn = ener_bond (bonds, bond_types, r_bond0, k_bond0)

    angles, angle_types, r_theta0, k_theta0 = \
             get_angles_info(prm_raw_data)
    ener_angle_fn = ener_angle (angles, angle_types, r_theta0, k_theta0)

    dihedrals, dihedral_types, dihedral_values = \
            get_dihedrals_info(prm_raw_data)
    ener_torsion_fn = ener_torsion (dihedrals, dihedral_types, dihedral_values)

    def compute_fn (R):

        enr_bond = ener_bond_fn (R)
        enr_angle = ener_angle_fn (R)
        enr_dih = ener_torsion_fn (R) 
        
        return enr_bond + enr_angle + enr_dih


    return compute_fn, ener_bond_fn



def nonbonded_LJ (dr, sigma, epsilon):
    '''
        dr, sigma, and epsilon are float
    '''
    idr = (sigma/dr)
    idr2 = idr*idr
    idr6 = idr2*idr2*idr2
    idr12 = idr6*idr6

    return jnp.nan_to_num(jnp.float32(4)*epsilon*(idr12-idr6))
    

def nonbonded_Coul (dr, chg_ij):
    
    return _ONE_4PI_EPS0*chg_ij/dr
    


def ener_nonbonded14 (atom_types, nonbonds, sigma, epsilon, chgs, 
                      vmax0=jnp.float32(50.0),
                      l_ewald=False,
                      eps_ewald=jnp.float32(1.0e-6), 
                      r_cut=jnp.float32(1.0)):
    """
    provide the nonbonded potential energy functions
    nonbonds: jnp.array ((n_pairs, 2), dtype=int) : provide pair_list
    nonbond_types: jnp.array (n_pairs, dtype=int) : index to atom_type_pairs
    sigma : jnp.array ( n_type_pairs, dtype=float)
    epsilon : jnp.array ( n_type_pairs, dtype=float)
    chgs : jnp.array (n_atom, dtype=float)
    """
    scee0 = jnp.float32(1.0/1.2)
    scnb0 = jnp.float32(1.0/2.0)

    at_type_a = atom_types[nonbonds[:,0]]
    at_type_b = atom_types[nonbonds[:,1]]
    sig_ab = 0.5*(sigma[at_type_a]+sigma[at_type_b])
    eps_ab = np.sqrt (epsilon[at_type_a]*epsilon[at_type_b])
    
    chg_ab =  chgs[nonbonds[:,0]]*chgs[nonbonds[:,1]] # chg_a (n_pair)
    U_chg0 = jax.vmap(nonbonded_Coul) (sig_ab, chg_ab)
    
    pp = -jnp.log (eps_ewald)
    aewald = jnp.sqrt (pp)/r_cut

    def compute_fn (R):
        # Ra and Rb (n_pairs, 3)
        
        Rab = R[nonbonds[:,1]] - R[nonbonds[:,0]]
        dr = jax.vmap(distance) (Rab)
        
        U_lj = scnb0*jax.vmap(nonbonded_LJ) (dr, sig_ab, eps_ab)
        U_chg = scee0*jax.vmap(nonbonded_Coul) (dr, chg_ab)
        U_lj = jnp.where (dr > sig_ab, U_lj, vmax0*jnp.tanh(U_lj/vmax0))
        U_chg = jnp.where (dr > sig_ab, U_chg, U_chg0+vmax0*jnp.tanh((U_chg-U_chg0)/vmax0))
        if l_ewald:
            alphaR = aewald*dr
            U_chg = U_chg * jax.scipy.special.erfc(alphaR)
        
        return jnp.sum(U_lj), jnp.sum(U_chg)

    return compute_fn



def ener_nonbonded_pair (atom_types, nonbonds, sigma, epsilon, chgs, 
                         vmax0=jnp.float32(50.0), 
                         l_ewald=False, 
                         switch=None, 
                         eps_ewald=jnp.float32(1.0e-6)):

    at_type_a = atom_types[nonbonds[:,0]]
    at_type_b = atom_types[nonbonds[:,1]]
    sig_ab = jnp.float32(0.5)*(sigma[at_type_a]+sigma[at_type_b])
    eps_ab = jnp.sqrt (epsilon[at_type_a]*epsilon[at_type_b])

    chg_ab = chgs[nonbonds[:,0]]*chgs[nonbonds[:,1]] # chg_a (n_pair)
    U_chg0 = jax.vmap(nonbonded_Coul) (sig_ab, chg_ab)

    cutoff_distance = jnp.float32 (1.0) # 10 A = 1 nm
    if switch is not None:
        switch_distance, cutoff_distance = switch
    
    pp = -jnp.log (eps_ewald)
    aewald = jnp.sqrt (pp)/cutoff_distance

    def compute_fn (R, Box=None): 
        
        Rab = R[nonbonds[:,1]] - R[nonbonds[:,0]]
        if Box is None:
            dr = jax.vmap(distance) (Rab)
        else:
            dr = jax.vmap(periodic_distance) (Rab)

        U_lj = jax.vmap(nonbonded_LJ) (dr, sig_ab, eps_ab)
        U_chg = jax.vmap (nonbonded_Coul) (dr, chg_ab)
        
        if switch is None:
            U_lj = jnp.where (dr > sig_ab, U_lj, vmax0*jnp.tanh(U_lj/vmax0))
            U_chg = jnp.where (dr > sig_ab, U_chg, U_chg0+vmax0*jnp.tanh((U_chg-U_chg0)/vmax0))
        else:
            tmp = (dr - switch_distance)/(cutoff_distance - switch_distance)
            svalue = 1 + tmp*tmp*tmp*(-10+tmp*(15-tmp*6))

            U_lj = jnp.where (dr > sig_ab, U_lj, vmax0*jnp.tanh(U_lj/vmax0))
            U_lj = jnp.where (dr < cutoff_distance, U_lj, jnp.float32(0.0))
            U_lj = jnp.where (dr < switch_distance, U_lj, U_lj*svalue)

            U_chg = jnp.where (dr > sig_ab, U_chg, U_chg0 + vmax0*jnp.tanh((U_chg-U_chg0)/vmax0))
            U_chg = jnp.where (dr < cutoff_distance, U_chg, jnp.float32(0.0))
            U_chg = jnp.where (dr < switch_distance, U_chg, U_chg*svalue)
            if l_ewald:
                alphaR = aewald*dr
                U_chg = U_chg * jax.scipy.special.erfc(alphaR)
        
        return jnp.sum(U_lj), jnp.sum(U_chg)

    return compute_fn


def get_amber_energy_funs (fname_prmtop):
    
    prm_raw_data = amber_prmtop_load (fname_prmtop)
    ener_bonded_fn, ener_bond_fn = ener_bonded (prm_raw_data)
    ener_bonded_fn = jax.jit (ener_bonded_fn)
    ener_bond_fn = jax.jit (ener_bond_fn)

    chgs = prm_get_charges (prm_raw_data)
    atom_types = prm_get_atom_types (prm_raw_data)
    sigma, epsilon = prm_get_nonbond_terms (prm_raw_data)
    
    nonbond_pairs = prm_get_nonbond_pairs (prm_raw_data)
    ener_nbond_fn = ener_nonbonded_pair (atom_types, nonbond_pairs,
                                        sigma, epsilon, chgs)
    ener_nbond_fn = jax.jit(ener_nbond_fn)

    nbonds14 = prm_get_nonbond14_info (prm_raw_data)
    ener_nbond14_fn = ener_nonbonded14 (atom_types, nbonds14,  
                                                sigma, epsilon, chgs)
    ener_nbond14_fn = jax.jit(ener_nbond14_fn)

    def compute_fun (R):
        """
        R (natom, 3)
        """
        en_bonded = ener_bonded_fn (R)
        en_lj, en_chg = ener_nbond_fn (R)
        en_lj14, en_chg14 = ener_nbond14_fn (R)
        
        return en_bonded + en_lj + en_chg + en_lj14 + en_chg14
    
    
    return compute_fun, ener_bond_fn, nonbond_pairs


if __name__ == '__main__':
    import mdtraj as md

    fname_dcd = 'test/L200/traj_complex_short.dcd'
    fname_prmtop = 'test/ala_deca_peptide.prmtop'

    c = md.load (fname_dcd, top=fname_prmtop)
    crds = jnp.array (c.xyz) # lenght unit is nm

    ener_fun, _, _ = get_amber_energy_funs (fname_prmtop)
    enr = jax.vmap (ener_fun) (crds)
    print ('<E>(kJ/mol) {:12.6f}'.format(enr.mean()))
    
