# LigandMPNN Docker/Apptainer

### Usage
- Docker: Add command
- Apptainer: Add command


### Implementation Notes
- In general, I try to avoid using conda inside containers for simplicity. However, in this case I can't get PRODY to be installed properly without it. Conda it is!
- Avoiding conda makes the apptainer def significantly simpler.

### To Do:
- Deleting symlinks still doesn't work...