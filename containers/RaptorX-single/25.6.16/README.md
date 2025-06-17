# RaptorX-Single Docker/Apptainer

Implementation of https://github.com/AndersJing/RaptorX-Single

### Usage
- Docker: ```python /app/RaptorX-Single/pred.py --plm_param_dir /app/RaptorX-Single/params/ test/ /app/RaptorX-Single/params/RaptorX-Single-ESM1b.pt  --device_id=0```
- Apptainer: ```apptainer run --nv raptorx.sif python /app/RaptorX-Single/pred.py ./test/IsPETase.fasta /app/RaptorX-Single/params/RaptorX-Single-ESM1b.pt --plm_param_dir /app/RaptorX-Single/params/ --out_dir ./out --device 0```



### Implementation Notes
- In general, I try to avoid using conda inside containers for simplicity. However, in this case I can't get PRODY to be installed properly without it. Conda it is!
- Avoiding conda makes the apptainer def significantly simpler.
- Source code location: `/app/RaptorX-Single/`
- Parameters: `/app/RaptorX-Single/params`
- micromamba environment: `/opt/conda/envs/app`


### To Do:
- Deleting symlinks still doesn't work...