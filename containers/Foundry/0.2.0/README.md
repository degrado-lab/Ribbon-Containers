# Foundry 0.2.0

Foundry is a suite of machine learning models for biomolecular design, structure prediction, and inverse folding developed by the Rosetta Commons and the Institute for Protein Design (IPD).

Included model suites:
- **RFdiffusion3 (RFD3)**: All-atom generative diffusion model for complex constrained design.
- **RFdiffusion3NA (RFD3NA)**: Nucleic acid and protein complex generative diffusion.
- **RoseTTAFold3 (RF3)**: All-atom structure prediction with implicit chirality and atom-level conditioning.
- **ProteinMPNN / LigandMPNN / SolubleMPNN**: Fast sequence design for fixed backbones and protein-ligand complexes.

All model weights (~7.9 GB) are baked directly into the container under `/weights`.

### Usage

#### Docker
```bash
# Interactive shell
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    nicholasfreitas/foundry:docker-0.2.0

# Run RF3 fold
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    nicholasfreitas/foundry:docker-0.2.0 \
    rf3 fold inputs=/workspace/test/5vht_single.json out_dir=/workspace/output_rf3

# Run RFD3 design
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    nicholasfreitas/foundry:docker-0.2.0 \
    rfd3 design inputs=/workspace/design_input.json out_dir=/workspace/output_rfd3

# Run MPNN sequence design
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    nicholasfreitas/foundry:docker-0.2.0 \
    mpnn --model_type protein_mpnn --structure_path /workspace/input.pdb --out_directory /workspace/mpnn_out
```

#### Apptainer / Singularity
```bash
# Interactive shell with GPU
apptainer shell --nv foundry_0.2.0.sif

# Run RF3 fold
apptainer run --nv foundry_0.2.0.sif \
    rf3 fold inputs=test/5vht_single.json out_dir=output_rf3

# Fully isolated offline run
apptainer run --nv --containall --net --network=none foundry_0.2.0.sif \
    rf3 fold inputs=test/5vht_single.json out_dir=/tmp/output_rf3

# Run MPNN
apptainer run --nv foundry_0.2.0.sif \
    mpnn --model_type protein_mpnn --structure_path input.pdb --out_directory mpnn_out
```

### Implementation Notes

1. **Base Image & Checkpoint Origin**:
   - Built on top of the official `rosettacommons/foundry:0.2.0-weights` release.
   - Pre-installed checkpoints in `/weights`:
     * `rf3_foundry_01_24_latest_remapped.ckpt` (2.83 GB)
     * `rfd3_latest.ckpt` (2.51 GB)
     * `rfd3na_1190.ckpt` (2.51 GB)
     * `proteinmpnn_v_48_020.pt` (6.7 MB)
     * `ligandmpnn_v_32_010_25.pt` (10.5 MB)
     * `solublempnn_v_48_020.pt` (6.7 MB)
2. **Runtime C Compiler (Triton / cuequivariance)**:
   - Upstream `rosettacommons/foundry:0.2.0-weights` omitted `gcc` and `build-essential` in its runtime image.
   - When executing `rf3 fold`, Triton's JIT CUDA driver (`CudaUtils` via `driver.c`) dynamically compiles modules using `$CC`. Without a C compiler, it crashes with `RuntimeError: Failed to find C compiler. Please specify via CC environment variable`.
   - We installed `build-essential` and `gcc` and exported `CC=/usr/bin/gcc` to enable Triton JIT compilation at runtime.
3. **Apptainer Read-Only Filesystem & Cache Redirection**:
   - Under Apptainer, images are mounted read-only.
   - Environment variables redirect all caches to `/tmp`:
     * `TRITON_CACHE_DIR=/tmp/triton`
     * `HF_HOME=/tmp/hf`
     * `XDG_CACHE_HOME=/tmp/cache`
     * `TORCHINDUCTOR_CACHE_DIR=/tmp/inductor`
     * `MPLCONFIGDIR=/tmp/mpl`
   - Offline enforcement is enabled with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
4. **Permissions**:
   - `/weights` and `/app` are set to `a+rX` so non-root users can read weights and execute scripts without permission errors.

### To Do
- [ ] Add optional ColabFold MSA database integration guidelines for large-scale MSA folding.
- [ ] Benchmark multi-GPU scaling on cluster configurations.
