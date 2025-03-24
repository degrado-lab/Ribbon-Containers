import argparse
import pymol
from pymol import cmd

def main():
    parser = argparse.ArgumentParser(
        description="Add hydrogens to a given residue or residues using PyMOL."
    )
    parser.add_argument(
        "pdb_file",
        help="Path to the structure file (e.g., PDB file) to load."
    )
    parser.add_argument(
        "--sel",
        default="all",
        help=("Residue selection for adding hydrogens. "
              "For example, use 'resi 50' or 'chain A and resi 50-60'. "
              "Defaults to 'all'.")
    )
    parser.add_argument(
        "--save",
        help=("Optional output filename to save the modified structure. "
              "If not provided, the structure will be modified but not saved.")
    )
    args = parser.parse_args()

    # Launch PyMOL in command-line mode (quiet and without GUI)
    pymol.finish_launching(['pymol', '-cq'])
    
    # Load the structure file
    cmd.load(args.pdb_file, 'mol')
    
    # Add hydrogens to the specified selection
    cmd.h_add(args.sel)
    
    # Save the modified structure if an output filename is provided
    if args.save:
        cmd.save(args.save, 'mol')
    
    # Exit PyMOL
    cmd.quit()

if __name__ == "__main__":
    main()
