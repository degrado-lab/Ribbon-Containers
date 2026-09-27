# BindCraft2 1.0.2

[BindCraft2](https://github.com/PacesaLab/BindCraft2) (BC2) is an automated, end-to-end pipeline for designing protein binders against target proteins developed by the Pacesa Lab. It combines sequence optimization through AlphaFold 2 with ProteinMPNN sequence redesign and structural validation filters.

Upstream Repository: [https://github.com/PacesaLab/BindCraft2](https://github.com/PacesaLab/BindCraft2)

---

### Usage

#### Docker
```bash
# Run design campaign
docker run --rm --gpus all -v $(pwd):/workspace nicholasfreitas/bindcraft2:docker-1.0.2 \
    bindcraft design examples/pdl1.json

# Design with custom settings and overrides
docker run --rm --gpus all -v $(pwd):/workspace nicholasfreitas/bindcraft2:docker-1.0.2 \
    bindcraft design examples/pdl1.json --modality VHH --humanize --set 'project_folder=results/pdl1_vhh'

# Rank designs by interface metric
docker run --rm -v $(pwd):/workspace nicholasfreitas/bindcraft2:docker-1.0.2 \
    bindcraft rank results/pdl1 --on i_pTM

# Filter rejected designs
docker run --rm -v $(pwd):/workspace nicholasfreitas/bindcraft2:docker-1.0.2 \
    bindcraft filter results/pdl1
```

#### Apptainer / Singularity
```bash
# Run design campaign with GPU and offline isolation
apptainer run --nv --containall --net --network=none bindcraft2_1.0.2.sif \
    bindcraft design /app/test/test_smoke.json

# Run custom campaign mounting current directory
apptainer run --nv --bind $(pwd):/workspace bindcraft2_1.0.2.sif \
    bindcraft design examples/pdl1.json

# Run realistic test with MPNN redesign & validation refolding
apptainer run --nv --containall --net --network=none \
    --bind $(pwd)/test:/workspace/test \
    bindcraft2_1.0.2.sif \
    bindcraft design /workspace/test/test_realistic.json --set 'project_folder=/workspace/test/output'

# Convert generated mmCIF (.cif) structures to standard PDB (.pdb)
apptainer exec --bind $(pwd)/test:/workspace/test bindcraft2_1.0.2.sif \
    python /workspace/test/cif2pdb.py /workspace/test/output

# Rank designs
apptainer run bindcraft2_1.0.2.sif \
    bindcraft rank results/pdl1 --on i_pTM

# Check environment and GPU devices
apptainer exec --nv bindcraft2_1.0.2.sif \
    python -c "import jax; print('JAX Backend:', jax.default_backend(), 'Devices:', jax.devices())"
```

---

### Implementation Notes

1. **Battery-Included Checkpoints & Weights**:
   - AlphaFold 2 parameters from Google DeepMind (`alphafold_params_2022-12-06.tar`) are baked directly into the image at `/app/models/alphafold/` (~5.3 GB).
   - All 7 campaign models required by BindCraft2 (`model_1_multimer_v3` through `model_5_multimer_v3`, `model_1_ptm`, `model_2_ptm`) are present and verified.
   - ProteinMPNN weights (`weights_neutral`, `weights_negative`, `weights_positive` for models `v_48_002`, `010`, `020`, `030`) are verified and stored at `/app/models/proteinmpnn/` and within the package tree.
   - A complete provenance and checksum manifest is generated at `/app/models/WEIGHTS.json`.

2. **JAX & Accelerator Architecture**:
   - Built on `nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba`.
   - Python 3.12 with JAX 0.11.2, jaxlib 0.11.2, `jax-cuda12-plugin`, and `cuequivariance-ops-cu12`.
   - Configured with `cuda12` to guarantee forward compatibility with CUDA 12 and CUDA 13 host drivers.
   - Dynamic linker configured via `/etc/ld.so.conf.d/bindcraft-cuda.conf` pointing to NVIDIA library paths inside site-packages.

3. **Apptainer Read-Only Filesystem & Cache Redirection**:
   - Under `--containall`, Apptainer mounts root as read-only.
   - JAX/XLA compilation cache (`JAX_COMPILATION_CACHE_DIR=/tmp/bindcraft_xla_cache`), HuggingFace (`HF_HOME=/tmp/hf`), XDG cache (`XDG_CACHE_HOME=/tmp/cache`), Triton (`TRITON_CACHE_DIR=/tmp/triton`), TorchInductor (`TORCHINDUCTOR_CACHE_DIR=/tmp/inductor`), and Matplotlib (`MPLCONFIGDIR=/tmp/mpl`) are redirected to `/tmp` in `%environment`.

4. **CLI Entrypoint Shims**:
   - Executable wrapper installed at both `/opt/conda/envs/app/bin/bindcraft` and `/usr/local/bin/bindcraft` executing `python -m bindcraft.cli "$@"`. This prevents option-swallowing by container activation shells.

5. **Self-Contained Validation**:
   - The `%test` block performs self-contained verification without GPU requirements, testing binary existence, checksums, package imports, and executing `bindcraft.selfcheck cuda12`.
   - Standalone GPU validation confirmed end-to-end design trajectory under `--containall --net --network=none`.

---

### To Do

- Benchmark JAX multi-GPU fan-out scaling across multiple nodes.
- Test optional ROCm / AMD GPU portability if requested.
