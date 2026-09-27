#!/usr/bin/env python3
"""
cif2pdb.py - Utility to convert BindCraft2 mmCIF (.cif) structures to standard PDB (.pdb) files.

Usage:
  python cif2pdb.py <path_or_directory>
"""

import os
import sys
from pathlib import Path
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx


def convert_cif_to_pdb(cif_path: Path) -> Path:
    pdb_path = cif_path.with_suffix('.pdb')
    try:
        cif_file = pdbx.CIFFile.read(str(cif_path))
        atoms = pdbx.get_structure(cif_file, model=1)
        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(atoms)
        pdb_file.write(str(pdb_path))
        print(f"Converted: {cif_path.name} -> {pdb_path.name} ({len(atoms)} atoms)")
        return pdb_path
    except Exception as e:
        print(f"Error converting {cif_path}: {e}", file=sys.stderr)
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python cif2pdb.py <cif_file_or_directory>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1]).resolve()
    if target.is_file() and target.suffix.lower() == '.cif':
        convert_cif_to_pdb(target)
    elif target.is_dir():
        cif_files = sorted(target.rglob("*.cif"))
        print(f"Found {len(cif_files)} .cif file(s) under {target}")
        converted = 0
        for cif in cif_files:
            if convert_cif_to_pdb(cif):
                converted += 1
        print(f"Successfully converted {converted}/{len(cif_files)} structures to PDB format.")
    else:
        print(f"Target not found or not a .cif/directory: {target}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
