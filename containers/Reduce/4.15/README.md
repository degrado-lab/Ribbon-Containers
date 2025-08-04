# RaptorX-Single Docker/Apptainer

Implementation of https://github.com/AndersJing/RaptorX-Single

### Usage
- Docker: ``` ```
- Apptainer: ```apptainer run --nv reduce_latest.sif reduce -build input.pdb > output.pdb```



### Implementation Notes
- Reduce works well for protein structures, but requires a ligand database TXT file for ligand protonation. 
- I haven't got this working yet for custom ligands, and may not unless the Reduce team gets back to me.


### To Do:
- Figure out custom ligands...