##########################################
### For use in the Biopython Container ###
##########################################

import sys
from Bio import PDB
import argparse

def calculate_atom_distance(pdb_file, atom1_info, atom2_info):
    # Parse the atom info strings in the format chain:residue:atom
    chain1_id, res1_id, atom1_name = atom1_info.split(":")
    chain2_id, res2_id, atom2_name = atom2_info.split(":")
    
    if pdb_file.endswith('.cif'):
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    chain1 = structure[0][chain1_id]
    chain2 = structure[0][chain2_id]
    res1 = get_residue(chain1, res1_id)
    res2 = get_residue(chain2, res2_id)
    atom1 = res1[atom1_name]
    atom2 = res2[atom2_name]
    distance = atom1 - atom2
    return distance

def get_residue(chain, res_id):
    """
    Try to retrieve the residue from the chain. First, try a standard residue,
    and if that fails, attempt to retrieve a hetero-residue.
    """
    try:
        # Try to get a standard residue (hetfield = ' ')
        residue = chain[(' ', int(res_id), ' ')]
    except KeyError:
        # If standard residue is not found, search through hetero-residues
        hetero_residues = [res for res in chain.get_list() if res.id[0] != ' ']
        for res in hetero_residues:
            if res.id[1] == int(res_id):
                return res
        raise KeyError(f"Residue {res_id} not found in chain {chain.id}")
    return residue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the distance between two atoms in a PDB file.")
    parser.add_argument("pdb_file", type=str, help="Path to the PDB file")
    parser.add_argument("atom1", type=str, help="First atom info in format chain:residue:atom")
    parser.add_argument("atom2", type=str, help="Second atom info in format chain:residue:atom")
    parser.add_argument("--output_file", type=str, help="Path to the output file", default=None)
    args = parser.parse_args()
    
    pdb_file = args.pdb_file
    atom1_info = args.atom1
    atom2_info = args.atom2
    output_file = args.output_file

    try:
        distance = calculate_atom_distance(pdb_file, atom1_info, atom2_info)
        # Re-parse for display
        chain1_id, res1_id, atom1_name = atom1_info.split(":")
        chain2_id, res2_id, atom2_name = atom2_info.split(":")
        print(f"The distance between {atom1_name} in residue {res1_id} (chain {chain1_id}) and {atom2_name} in residue {res2_id} (chain {chain2_id}) is: {distance:.2f} Å")
        if output_file:
            with open(output_file, "w") as f:
                f.write(f"{distance:.2f}")
    except KeyError as e:
        print(f"Error: Residue, chain, or atom not found. {e}")