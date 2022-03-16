import numpy as np
import jax.numpy as jnp
import jax
import jax_md
import re

def amber_parmtop_load (fname_prmtop):
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
    sigma_scale = 2.0**(1.0/6.0)
    numTypes_2 = numTypes*(numTypes+1)//2
    sigma_AB = np.zeros ( numTypes_2 )
    epsilon_AB = np.zeros ( numTypes_2 )

    for i in range(numTypes):
        for j in range(i+1):
            index = int (prm_raw_data['NONBONDED_PARM_INDEX'][numTypes*i+j]) - 1
            if index < 0: continue

            index_2 = i*(i+1)//2 + j
            acoef = float(LJ_ACOEF[index])
            bcoef = float(LJ_BCOEF[index])

            try:
                sigma = (acoef/bcoef)**(1.0/6.0)
                epsilon = 0.25*bcoef*bcoef/acoef
            except ZeroDivisionError:
                sigma = 1.0
                epsilon = 0.0

            sigma_AB[index_2] = sigma*lengthConversionFactor
            epsilon_AB[index_2] = epsilon*energyConversionFactor

    return jnp.array(sigma_AB), jnp.array(epsilon_AB)


def prm_get_nonbond_info (prm_raw_data):

    def get_mask_nonbond_pairs ():
        num_excl_atoms = prm_raw_data['NUMBER_EXCLUDED_ATOMS']
        excl_atoms_list = prm_raw_data['EXCLUDED_ATOMS_LIST']
        total = 0
        numAtoms = int(prm_raw_data['POINTERS'][0])
        mask_nonbond_pairs = np.ones ( (numAtoms, numAtoms), dtype=int) == 1
        
        for iatom in range(numAtoms):
            index0 = total
            n = int (num_excl_atoms[iatom])
            total += n
            index1 = total
            for jatom in excl_atoms_list[index0:index1]:
                j = int(jatom) - 1
                mask_nonbond_pairs[iatom, j] = False
                mask_nonbond_pairs[j, iatom] = False
            mask_nonbond_pairs[iatom, iatom] = False

        return mask_nonbond_pairs

    atom_types = prm_get_atom_types (prm_raw_data)
    mask_nonbond_pairs = get_mask_nonbond_pairs ()

    numAtoms = int(prm_raw_data['POINTERS'][0])
    atom_idx = np.arange(numAtoms)
    nonbond_pairs = []
    nonbond_types = []
        
    for iatom in range (numAtoms):
        at_i = atom_types[iatom]
        jatom_idx = atom_idx[mask_nonbond_pairs[iatom]]
        mask_jatom = jatom_idx < iatom

        for jatom in jatom_idx[mask_jatom]:
            at_j = atom_types[jatom]
            nonbond_pairs.append ( (iatom, jatom) )

            if at_i >= at_j:
                idx_2 = at_i*(at_i+1)//2 + at_j
            else:
                idx_2 = at_j*(at_j+1)//2 + at_i
                
            nonbond_types.append (idx_2)
        
    return jnp.array(nonbond_pairs), jnp.array(nonbond_types)


def prm_get_nonbond14_info (prm_raw_data):
    
    atom_types = prm_get_atom_types (prm_raw_data)
    dihedralPointers = prm_raw_data["DIHEDRALS_WITHOUT_HYDROGEN"] + \
                            prm_raw_data["DIHEDRALS_INC_HYDROGEN"] 

    nonbond14_pairs = []
    nonbond14_types = []

    for ii in range (0, len(dihedralPointers), 5):
        if int(dihedralPointers[ii+2])>0 and int(dihedralPointers[ii+3])>0:
            iAtom = int(dihedralPointers[ii])//3
            lAtom = int(dihedralPointers[ii+3])//3
            
            nonbond14_pairs.append ( (iAtom, lAtom) )

            at_i = atom_types[iAtom]
            at_l = atom_types[lAtom]

            if at_i >= at_l:
                idx_2 = at_i*(at_i+1)//2 + at_l
            else:
                idx_2 = at_l*(at_l+1)//2 + at_i 

            nonbond14_types.append (idx_2)

    return jnp.array(nonbond14_pairs), jnp.array(nonbond14_types)


def ener_bond (bonds, bond_types, rb0, kb0):
    '''
    bonds : jnp.array ( (n_bond, 2), dtype=int )
    bond_types: jnp.array ( (n_bond), dtype=int )
    rb0 : jnp.array ( (n_bond_type), dtype=float )
    kb0 : jnp.array ( (n_bond_type), dtype=float )
    '''
    r0 = jax_md.util.maybe_downcast (rb0[bond_types])
    k0 = jax_md.util.maybe_downcast (kb0[bond_types])

    def _bond_interaction (r):
        # U = 0.5 * k0 * (r - r0)^2
        return jnp.float32(0.5)*k0*(r - r0)**2

    def distance (R1, R2):
        # R1, R2 (3)
        dR = R2 - R1
        return jnp.sqrt (jnp.einsum('i,i->', dR, dR))
        

    def compute_fn (R):
        '''
        R ((n_atom, 3),dtype=float)
        '''
        R1 = R[bonds[:,0]] # (n_bond, 3)
        R2 = R[bonds[:,1]]

        # r (n_bond)
        r  = jax.vmap(distance) (R1, R2)
        # en_val (n_bond)
        en_val = _bond_interaction (r)

        return jax_md.util.high_precision_sum (en_val)

    return compute_fn


def ener_angle (angles, angle_types, r_theta0, k_theta0):
    '''
    angles: jnp.array ((n_angle, 3), dtype=int) 
    angle_types: jnp.array ( (n_angle), dtype=int )
    r_theta0 : jnp.array ( (n_angle_type), dtype=float )
    k_theta0 : jnp.array ( (n_angle_torsion), dtype=float )
    '''
    theta0 = jax_md.util.maybe_downcast (r_theta0[angle_types])
    k0 = jax_md.util.maybe_downcast (k_theta0[angle_types])

    def _angle_interaction (theta):
        # U = 0.5* k0 * (theta - theta0)^2
        return jnp.float32(0.5)*k0*(theta-theta0)**2

    def theta_fn (R1, R2, R3):
        '''
        R1, R2, R3 : jnp.array((3), dtype=float)
        '''
        dR21 = R1 - R2
        dR23 = R3 - R2

        cos_angle = jax_md.quantity.cosine_angle_between_two_vectors(dR21, dR23)
        return jnp.arccos (cos_angle)
    
    def compute_fn (R):
        '''
        R: jnp.array ( (n_atom, 3), dtype=float)
        '''
        R1 = R[angles[:, 0]] # (n_angle, 3)
        R2 = R[angles[:, 1]]
        R3 = R[angles[:, 2]]

        # theta (n_angle)
        theta = jax.vmap (theta_fn) (R1, R2, R3)
        # en_val (n_angle)
        en_val = _angle_interaction (theta)

        return jax_md.util.high_precision_sum (en_val)

    return compute_fn


def ener_torsion (torsions, torsion_types, torsion_values):
    '''
    torsions: jnp.array ((n_torsion, 4), dtype=int) 
    torsion_types: jnp.array ( (n_torsion), dtype=int )
    n_theta0 : jnp.array ( (n_torsion_type), dtype=int )
    cos_phase0 : jnp.array ( (n_torsion_type), dtype=float )
    k_theta0 : jnp.array ( (n_torsion_torsion), dtype=float )
    '''
    n_theta0, cos_phase0, k_theta0 = torsion_values
    n0 = jax_md.util.maybe_downcast (n_theta0[torsion_types])
    cos_phase = jax_md.util.maybe_downcast (cos_phase0[torsion_types])
    k0 = jax_md.util.maybe_downcast (k_theta0[torsion_types])

    def _torsion_interaction (theta):
        # theta : torsional angle
        # cos_phase0 = cos (theta0), where theta0 is 0 or pi
        # U_torsion = k0 * (1 + cos (n0 theta - theta0))
        # U_torsion = k0 * (1 + cos (n0 theta) * cos_phase0)

        return k0*(1.0 + jnp.cos(n0*theta)*cos_phase)


    def theta_fn (R1, R2, R3, R4):
        '''
        Estimate torsional angle using four atoms: R1, R2, R3, R4
        R1, R2, R3, R4: jnp.array ( (3), dtype=float)
        '''
        dR12 = R2 - R1
        dR23 = R3 - R2
        dR34 = R4 - R3
        dRT = jnp.cross (dR12, dR23)
        dRU = jnp.cross (dR23, dR34)
        dTU = jnp.cross (dRT, dRU)

        rt = jax_md.space.distance (dRT) + 1.e-7
        ru = jax_md.space.distance (dRU) + 1.e-7
        r23 = jax_md.space.distance (dR23) + 1.e-7

        cos_angle = jnp.dot (dRT, dRU)/(rt*ru)
        sin_angle = jnp.dot (dR23, dTU)/(r23*rt*ru)
        theta = jnp.arctan2(sin_angle, cos_angle)

        return theta



    def compute_fn (R):
        '''
        R: jnp.array( (n_atom, 3), dtype=float)
        '''
        R1 = R[torsions[:,0]] # R1 (n_torsion, 3)
        R2 = R[torsions[:,1]]
        R3 = R[torsions[:,2]]
        R4 = R[torsions[:,3]]

        # theta (n_torsion)
        theta = jax.vmap (theta_fn) (R1, R2, R3, R4)
        # en_val (n_torsion)
        en_val = _torsion_interaction (theta)

        return jax_md.util.high_precision_sum(en_val)
    
    return compute_fn




def ener_bonded (prm_raw_data):
    '''
    prmtop._raw_data: dict{} : from amber prmtop file
    ener_bonded = ener_bond + ener_angle + ener_torsion
    '''

    def get_bonds_info ():
        forceConstant = []
        for k0 in prm_raw_data['BOND_FORCE_CONSTANT']:
            forceConstant.append (float(k0))
        
        bondEquil = []
        for r0 in prm_raw_data['BOND_EQUIL_VALUE']:
            bondEquil.append (float(r0))
        
        # kcal/mol/A^2 --> kJ/mol/nm^2
        forceConstConversionFactor = jnp.float32 (418.4) 
        # A --> nm
        lengthConversionFactor = jnp.float32 (0.1)
        # Amber : k(r - r0)^2
        # openmm and this code : 0.5 * k' (r - r0)^2
        # k' = 2 * k
        forceConstant = jnp.float32(2.0)*jnp.array (forceConstant)*forceConstConversionFactor
        bondEquil = jnp.array(bondEquil)*lengthConversionFactor

        bondPointers = prm_raw_data['BONDS_WITHOUT_HYDROGEN'] + \
            prm_raw_data['BONDS_INC_HYDROGEN']
        
        bonds = []
        bond_types = []
        for ii in range (0, len(bondPointers), 3):
            iType = int (bondPointers[ii+2]) - 1
            bonds.append ( (int(bondPointers[ii])//3,
                            int(bondPointers[ii+1])//3) )
            bond_types.append(iType)

        return jnp.array(bonds), jnp.array(bond_types), bondEquil, forceConstant


    def get_angles_info ():
        forceConstant = []
        for k0 in prm_raw_data['ANGLE_FORCE_CONSTANT']:
            forceConstant.append (float(k0))
        
        angleEquil = []
        for r0 in prm_raw_data['ANGLE_EQUIL_VALUE']:
            angleEquil.append (float(r0))
        
        # kcal/mol/rad^2 --> kJ/mol/rad^2
        forceConstConversionFactor = jnp.float32 (4.184) 
        
        # Amber : k(r - r0)^2
        # openmm and this code : 0.5 * k' (r - r0)^2
        # k' = 2 * k
        forceConstant = jnp.float32(2.0)*jnp.array (forceConstant)*forceConstConversionFactor
        angleEquil = jnp.array(angleEquil)

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

        return jnp.array(angles), jnp.array(angle_types), angleEquil, forceConstant


    def get_dihedrals_info ():
        forceConstant = []
        for k0 in prm_raw_data['DIHEDRAL_FORCE_CONSTANT']:
            forceConstant.append (float(k0))
        
        cos_phase0 = []
        for ph0 in prm_raw_data['DIHEDRAL_PHASE']:
            val = np.cos (float(ph0))
            if val < 0:
                cos_phase0.append (-1.0)
            else:
                cos_phase0.append (1.0)
        
        periodicity = []
        for n0 in prm_raw_data['DIHEDRAL_PERIODICITY']:
            periodicity.append (int(0.5 + float(n0)))
        

        # kcal/mol/rad^2 --> kJ/mol/rad^2
        forceConstConversionFactor = jnp.float32 (4.184) 
        
        
        forceConstant = jnp.array (forceConstant)*forceConstConversionFactor
        cos_phase0 = jnp.array(cos_phase0)
        periodicity = jnp.array(periodicity)


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
        return jnp.array(dihedrals), jnp.array(dihedral_types), dihedral_type_values
    

    def bond_energy_fn (R):
        # Bond
        bonds, bond_types, r0, k0 = get_bonds_info()
        return ener_bond (bonds, bond_types, r0, k0) (R)


    def angle_energy_fn (R):
        # Angle
        angles, angle_types, r_theta0, k_theta0 = get_angles_info()
        return ener_angle (angles, angle_types, r_theta0, k_theta0) (R)


    def torsion_energy_fn (R):
        # Dihedral
        dihedrals, dihedral_types, dihedral_values = get_dihedrals_info()
        return ener_torsion (dihedrals, dihedral_types, dihedral_values) (R)


    def compute_fn (R):
    
        return bond_energy_fn(R) + angle_energy_fn (R) + torsion_energy_fn (R)

    return compute_fn, bond_energy_fn


def ener_nonbonded (nonbonds, nonbond_types, sigma, epsilon, chgs, scee0, scnb0):
    """
    provide the nonbonded potential energy functions
    nonbonds: jnp.array ((n_pairs, 2), dtype=int) : provide pair_list
    nonbond_types: jnp.array (n_pairs, dtype=int) : index to atom_type_pairs
    sigma : jnp.array ( n_type_pairs, dtype=float)
    epsilon : jnp.array ( n_type_pairs, dtype=float)
    chgs : jnp.array (n_atom, dtype=float)
    """

    sig_ab = jax_md.util.maybe_downcast (sigma[nonbond_types]) # sig_ab (n_pair)
    eps_ab = jax_md.util.maybe_downcast (epsilon[nonbond_types])
    chg_a  = jax_md.util.maybe_downcast (chgs[nonbonds[:,0]]) # chg_a (n_pair)
    chg_b  = jax_md.util.maybe_downcast (chgs[nonbonds[:,1]])

    def nonbonded_LJ (dr, sigma, epsilon):
        '''
        dr, sigma, and epsilon are float
        '''
        idr = (sigma/dr)
        idr2 = idr*idr
        idr6 = idr2*idr2*idr2
        idr12 = idr6*idr6

        return jnp.nan_to_num(jnp.float32(4)*epsilon*(idr12-idr6))


    def nonbonded_Coul (dr, chg_i, chg_j):
        _ONE_4PI_EPS0 = jnp.float32(138.935456)
        return _ONE_4PI_EPS0*chg_i*chg_j/dr

        
    def compute_fn (R):
        # Ra and Rb (n_pairs, 3)
        Ra = R[nonbonds[:,0]]
        Rb = R[nonbonds[:,1]]
        Rab= Rb - Ra
        dr = jax_md.space.distance (Rab)
            
        U_lj = scnb0*jax.vmap(nonbonded_LJ) (dr, sig_ab, eps_ab)
        U_chg = scee0*jax.vmap(nonbonded_Coul) (dr, chg_a, chg_b)
            
        return jax_md.util.high_precision_sum(U_lj), jax_md.util.high_precision_sum(U_chg)

    return compute_fn


if __name__ == '__main__':
    import MDAnalysis as mda

    fname_prmtop = 'da.prmtop'
    fname_dcd = 'traj_complex.dcd'
    fname_pdb = 'da_nvt.pdb'

    prm_raw_data = amber_parmtop_load(fname_prmtop)
    ener_bonded_fn, ener_bond_fn = ener_bonded (prm_raw_data)
    ener_bonded_fn = jax.jit (ener_bonded_fn)

    chgs = prm_get_charges (prm_raw_data)
    sigma, epsilon = prm_get_nonbond_terms (prm_raw_data)
    nbond_pairs, nbond_types = prm_get_nonbond_info (prm_raw_data)
    nbond14_pairs, nbond14_types = prm_get_nonbond14_info (prm_raw_data)
    scee0  = jnp.float32(1.0)
    scnb0  = jnp.float32(1.0)
    ener_nonbonded_fn = ener_nonbonded (nbond_pairs, nbond_types, 
                                    sigma, epsilon, chgs, scee0, scnb0)
        
    scee0 = jnp.float32(1.0/1.2)
    scnb0 = jnp.float32(1.0/2.0)
    ener_nonbonded14_fn = ener_nonbonded (nbond14_pairs, nbond14_types, 
                                    sigma, epsilon, chgs, scee0, scnb0)

    ener_nonbonded_fn = jax.jit(ener_nonbonded_fn)
    ener_nonbonded14_fn = jax.jit(ener_nonbonded14_fn)

    l_PDB = True

    if l_PDB:
        u = mda.Universe(fname_pdb)
        x_Ai = jnp.array(u.atoms.positions)*jnp.float32(0.1) # A-->nm
        
        en_bonded = ener_bonded_fn (x_Ai)
        en_bond = ener_bond_fn (x_Ai)
        print ('en_bond', en_bond/4.184)
        print ('en_bonded', en_bonded/4.184)

        (en_lj, en_chg) = ener_nonbonded_fn (x_Ai)
        (en_lj14, en_chg14) = ener_nonbonded14_fn (x_Ai)
        print ('en_nbond', en_lj/4.184, en_chg/4.184)
        print ('en_nbond14', en_lj14/4.184, en_chg14/4.184)
    else:
        u = mda.Universe(fname_prmtop, fname_dcd)
        x_Ai = []
        for ts in u.trajectory:
            crds = u.atoms.positions
            x_Ai.append(crds)
    
        x_Ai = jnp.array(x_Ai[-1000:])*jnp.float32(0.1) # A --> nm
        en_bonded = jax.vmap(ener_bonded_fn) (x_Ai)
        en_bond = jax.vmap(ener_bond_fn) (x_Ai)
        print ('en_bond', en_bond[:5]/4.184)
        print ('en_bonded', en_bonded[:5]/4.184)

        (en_lj, en_chg) = jax.vmap(ener_nonbonded_fn) (x_Ai)
        (en_lj14, en_chg14) = jax.vmap(ener_nonbonded14_fn) (x_Ai)
        print ('en_nbond', en_lj[:5]/4.184, en_chg[:5]/4.184)
        print ('en_nbond14', en_lj14[:5]/4.184, en_chg14[:5]/4.184)