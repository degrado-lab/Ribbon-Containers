# RaptorX-Single Docker/Apptainer

Implementation of https://github.com/AndersJing/RaptorX-Single

### Usage
- Docker: ```docker run --gpus all -v .:/opt/RaptorX-Single/working nicholasfreitas/raptorx-single:docker-3.1.25 python pred.py ./working/test_data/ params/RaptorX-Single-ESM1b.pt --out_dir ./working/out4 --device_id 0 ```
- Apptainer: ```apptainer run --nv raptorx-single_3.1.25.sif conda run -n RaptorX-Single python RaptorX-Single/pred.py test_data/HG3.fasta RaptorX-Single/params/RaptorX-Single-ESM1b.pt --plm_param_dir RaptorX-Single/params/ --out_dir ./out --device 0```



### Implementation Notes
- In general, I try to avoid using conda inside containers for simplicity. However, in this case I can't get PRODY to be installed properly without it. Conda it is!
- Avoiding conda makes the apptainer def significantly simpler.
- Source code location: `/opt/RaptorX-Single/`
- Parameters: `/opt/RaptorX-Single/params`
- Conda environment: `/usr/envs/RaptorX-Single/`
- For some reason, the apptainer version doesn't find the parameters directory correctly. I suspect this has to do with the fact that it's using environment variables to store the location, and maybe those work differently in the apptainer? Not sure, but for now I'm creaiting and killing a symlink like I did with LigandMPNN. (This is not ideal, but fine for now.)


### To Do:
- Deleting symlinks still doesn't work...