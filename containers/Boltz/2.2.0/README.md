# LigandMPNN Docker/Apptainer

### Usage
- Docker: `docker run -it --gpus all --shm-size=2g -v .:/workspace boltz:test`
- Apptainer: `apptainer run --nv boltz.sif boltz predict test.yml`


### Implementation Notes
- I've set $BOLTZ_CACHE to /app/boltz, where all the weights are downloaded.
- Currently, only Boltz-2 weights are downloaded by default. (Should we offer Boltz-1 weights too?)
- Download script 'download_weights.py' is used (can be re-used for Boltz-1)
- Preparing MSAs example is here: https://github.com/jwohlwend/boltz/issues/6


### To Do:
- Sort out Boltz-1 weights?