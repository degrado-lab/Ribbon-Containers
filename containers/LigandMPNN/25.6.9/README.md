# LigandMPNN Docker/Apptainer

### Usage
- Docker: Add command
- Apptainer: `apptainer run --nv ligandmpnn.sif python LigandMPNN/run.py --pdb_path test_data/HG3.pdb --out_folder output/`


### Implementation Notes
- In general, I try to avoid using conda inside containers for simplicity. However, in this case I can't get PRODY to be installed properly without it. Conda it is!
- Avoiding conda makes the apptainer def significantly simpler.

### To Do:
- Deleting symlinks still doesn't work...