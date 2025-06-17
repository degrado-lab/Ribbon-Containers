# ESM3 Docker/Apptainer

### Usage
- Docker: Add command
- Apptainer: `apptainer run --nv ligandmpnn.sif python LigandMPNN/run.py --pdb_path test_data/HG3.pdb --out_folder output/`


### Implementation Notes
- ESM has various models available in the cloud API.
- **IMPORTANT** The token thing may prove to be too much of a headache. It looks like it has to both read and write to the weights directory at runtime, which is a no-go with apptainer's read-only filesystem. I'll have the user handle downloads as usual, with only a minimal install. That's for later, though.
- Since ESM3 is a restricted model (no consumer use), we must pass in a token or hard-code it in a script with `login(token="[token]")`. Our DeGrado Lab token is in the DGL Wiki and is **private**. It should never be shared with anyone outside the lab, or published in any public repository.


### To Do:
- None.