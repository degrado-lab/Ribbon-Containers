# LDDM 0.1.0

[LDDM](https://github.com/LPDI-EPFL/lddm) (Large Drug Discovery Model) is a unified 3D generative model for structure-based drug design developed by the Laboratory of Protein Design & Immunoengineering (LPDI) at EPFL. It supports de novo design, fragment growing/linking, unconstrained docking, partial/constrained docking, and programmable/synthesizable ligand generation in virtual chemical spaces.

Academic Preprint: [bioRxiv: 10.64898/2026.09.15.751537v1](https://www.biorxiv.org/content/10.64898/2026.09.15.751537v1)

---

### Usage

#### Docker
```bash
# De novo design
docker run --rm --gpus all -v $(pwd):/workspace nicholasfreitas/lddm:docker-0.1.0 \
    lddm-sample design \
        --protein examples/kras.pdb \
        --ref_ligand examples/kras_ref_ligand.sdf \
        --checkpoint /app/models/checkpoints/lddm.ckpt \
        --output out/de_novo.sdf \
        --n_samples 10

# Molecular docking
docker run --rm --gpus all -v $(pwd):/workspace nicholasfreitas/lddm:docker-0.1.0 \
    lddm-sample dock \
        --protein examples/kras.pdb \
        --ref_ligand examples/kras_ref_ligand.sdf \
        --ligand examples/kras_ref_ligand.sdf \
        --checkpoint /app/models/checkpoints/lddm.ckpt \
        --output out/docked.sdf

# Programmable design
docker run --rm --gpus all -v $(pwd):/workspace nicholasfreitas/lddm:docker-0.1.0 \
    lddm-programmable /app/configs/programmable_design.yml \
        --protein examples/kras.pdb \
        --ligand examples/kras_ref_ligand.sdf \
        --output out/kras_prog
```

#### Apptainer / Singularity
```bash
# De novo design
apptainer run --nv lddm_0.1.0.sif \
    lddm-sample design \
        --protein /app/test/kras.pdb \
        --ref_ligand /app/test/kras_ref_ligand.sdf \
        --checkpoint /app/models/checkpoints/lddm.ckpt \
        --output out/de_novo.sdf \
        --n_samples 10

# Molecular docking
apptainer run --nv lddm_0.1.0.sif \
    lddm-sample dock \
        --protein /app/test/kras.pdb \
        --ref_ligand /app/test/kras_ref_ligand.sdf \
        --ligand /app/test/kras_ref_ligand.sdf \
        --checkpoint /app/models/checkpoints/lddm.ckpt \
        --output out/docked.sdf

# Programmable design with precompiled validity references
apptainer run --nv lddm_0.1.0.sif \
    lddm-programmable /app/configs/programmable_design.yml \
        --protein /app/test/kras.pdb \
        --ligand /app/test/kras_ref_ligand.sdf \
        --output out/kras_prog
```

---

### Implementation Notes

1. **Battery-Included Checkpoints**:
   - Both published checkpoints from Zenodo ([Record 22754501](https://zenodo.org/records/22754501)) are baked into the image at `/app/models/checkpoints/`:
     - `lddm.ckpt`: Primary model trained on BindingNet (`CC-BY-NC 4.0`), used in the paper.
     - `lddm_CDBB.ckpt`: Model trained without BindingNet (`MIT License`).
   - Provenance and checksums are stored in `/app/models/WEIGHTS.json`.

2. **Precompiled 3D Validity Geometry Cache**:
   - Upstream compiles reference KDE distributions from `ligands.sdf` on first run, writing into the dataset directory.
   - Because Apptainer mounts `.sif` files read-only, attempting to compute and write cache files at runtime causes `[Errno 30] Read-only file system`.
   - To make the container 100% self-contained and read-only safe, `ligands_geometry_values.p` and `ligands_geometry_kernel_densities.p` are precompiled at build time into `/app/data/validity3d/`.

3. **External Scientific Toolchains**:
   - **Reduce v4.15**: Compiled from source and installed to `/opt/conda/envs/app/bin/reduce` and `/usr/local/bin/reduce`. Required for protonation in `interactions_local` evaluations.
   - **Gnina v1.1**: Standalone binary installed to `/opt/conda/envs/app/bin/gnina` and `/usr/local/bin/gnina`. Required for affinity scoring and minimization.

4. **Environment & Cache Redirection**:
   - Built on `nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba`.
   - Python 3.11 with PyTorch 2.6.0+cu124, PyG 2.7.0, and torch-scatter 2.1.2.
   - Caches for Hugging Face, Torch, Triton, TorchInductor, and Matplotlib are redirected to `/tmp` in `%environment` to guarantee execution under `--containall`.

5. **PATH Entrypoint Shims**:
   - Installed on `$PATH` to prevent option-swallowing by micromamba wrappers:
     - `lddm-sample`: runs `scripts/sample.py`
     - `lddm-programmable`: runs `scripts/generate_programmable_design.py`
     - `lddm-prep-chemspace`: runs `scripts/prepare_chemical_space.py`
     - `lddm-prep-synspace`: runs `scripts/prepare_synspace.py`

---

### To Do

- Benchmark synthesizable design speed across varying chemical space sizes.
- Evaluate upgrading Gnina to v1.3 once a compatible standalone build is verified.
