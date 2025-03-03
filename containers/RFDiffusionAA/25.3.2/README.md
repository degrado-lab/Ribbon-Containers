# LigandMPNN Docker/Apptainer

https://github.com/baker-laboratory/rf_diffusion_all_atom

### Usage
- Docker: Not implemented.
- Apptainer: ``` apptainer run --nv rfdiffusionaa_25.3.2.sif python -u rf_diffusion_all_atom/run_inference.py inference.input_pdb=rf_diffusion_all_atom/input/7v11.pdb contigmap.contigs=[\'150-150\'] inference.ligand=OQO inference.num_designs=1```


### Implementation Notes
- This is based on the existing apptainer image (`rf_se3_diffusion.sif`) downloaded with:
    - `wget http://files.ipd.uw.edu/pub/RF-All-Atom/containers/rf_se3_diffusion.sif`
- I pre-download the weights (since apptainer doesn't use nice layering like Docker). Downloaded from:
    - `wget http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFDiffusionAA_paper_weights.pt`

Because of paths in the code, I needed to add a symlink to the paper weights in the current directory. Additionally, I needed to copy the full RFDiffusion code into the current directory (under `./rf_diffusion_all_atom/`). This is because at runtime, the code caches files within the code directory (this is hard-coded), and apptainer is a read-only system. In the future I'd like to clean this up.

### To Do:
- Rebuild to add final line where the copied `./rf_diffusion_all_atom/` directory is deleted.
- Potentially fork code to  clean up paths etc. Low priority.