
from Bio import PDB
import numpy as np


parser = PDB.PDBParser()
io = PDB.PDBIO()


def get_chains(target_id, pdb_file):
    structure = parser.get_structure(target_id, pdb_file)
    model=structure[0]
    
    chain1 = model[target_id.split('_')[0][-1]]
    chain2 = model[target_id.split('_')[1][-1]]


    return chain1, chain2



def get_ca_from_chain(chain):
    ca_atoms = []
    
    for residue in chain:
        ca_atoms.append(residue['CA'].get_coord())

    return np.array(ca_atoms)



def get_atoms_from_chain(chain):
    all_atoms = []
    
    for residue in chain:
        for atom in residue:
            all_atoms.append(atom.get_coord())

    return np.array(all_atoms)


def get_CA_atoms(target_id, pdb_file):
    chain1, chain2 = get_chains(target_id, pdb_file)

    all_ca_chain1 = get_ca_from_chain(chain1)
    all_ca_chain2 = get_ca_from_chain(chain2)

    all_ca_atoms = np.concatenate((all_ca_chain1, all_ca_chain2), axis=0)

    return all_ca_atoms




def compute_RMSD(set1, set2):
    assert set1.shape[0] == set2.shape[0]
    L = set1.shape[0]
    assert L > 0

    COM1 = np.sum(set1, axis=0) / float(L)
    COM2 = np.sum(set2, axis=0) / float(L)

    

    set1 -= COM1
    set2 -= COM2

    E0 = np.sum(np.sum(set1*set1, axis=0), axis=0) + np.sum(np.sum(set2*set2, axis=0), axis=0)

    V, S, Wt = np.linalg.svd(np.dot(np.transpose(set2), set1))

    reflect = float(str(float(np.linalg.det(V) * np.linalg.det(Wt))))
    
    if reflect == -1.0:
        S[-1] = - S[-1]
        V[:, -1] = -V[:, -1]

    RMSD = E0 - (2.0 * sum(S))
    RMSD = np.sqrt(abs(RMSD/L))

    return RMSD






def write_pdb(pdb_file, new_coordinates, out_pdb):
    lines = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                lines.append(line)


    
    new_lines = []
    for i in range(len(lines)):
        x = str(format(new_coordinates[i][0], '.3f'))
        y = str(format(new_coordinates[i][1], '.3f'))
        z = str(format(new_coordinates[i][2], '.3f'))
        new_line = lines[i][0:30] + ' ' * (7-len(x)) + str(x) + ' ' + ' ' * (7-len(y)) + str(y) + ' ' + ' ' * (7-len(z)) + str(z) + lines[i][54:]
        
        new_lines.append(new_line)

    with open(out_pdb, 'w') as f:
        f.writelines(new_lines)
        
        

    


#write_pdb(f'{target_id}.pdb', all_atoms, f'{target_id}_modified.pdb')



