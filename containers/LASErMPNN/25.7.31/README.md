# LASErMPNN Docker/Apptainer

### Usage
- Docker: Add command
- Apptainer: `apptainer run --nv lasermpnn.sif python -m LASErMPNN.run_inference input.pdb --output_path output`


### Implementation Notes
- Can switch to run_batch_inference for many predictions.
- Need to add info on the many different input parameters...

### To Do:
- Deleting symlinks still doesn't work...