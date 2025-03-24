# EasyMD Docker/Apptainer

### Usage
- Docker: Add command
- Apptainer: Add command


### Implementation Notes
- In general, I try to avoid using conda inside containers for simplicity. However, OpenMM has several conda-specific requirements.
- The conda solve was so slow with the default environment.yml, it would time out in the Docker build. So, I added the more explicit environment.yml by exporting my pre-built environment.

### To Do:
