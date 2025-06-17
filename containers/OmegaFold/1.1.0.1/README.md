# OmegaFold Docker/Apptainer

### Usage
- Docker: Add command
- Apptainer: ```apptainer run --nv omegafold.sif omegafold --weights_file /app/model.pt test/IsPETase.fasta out/```


### Implementation Notes
- Weights are stored at `/app/model.pt`, and should be passed in using `--weights_file`. (Otherwise, the model will download them).

### To Do:
- 