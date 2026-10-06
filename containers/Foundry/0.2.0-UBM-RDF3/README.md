# Foundry 0.2.0-UBM-RDF3

Foundry is a suite of machine learning models for biomolecular design, structure prediction, and inverse folding developed by the Rosetta Commons and the Institute for Protein Design (IPD).

This container release (**0.2.0-UBM-RDF3**) incorporates Anthropic's **Uplifting Biomolecular Modeling (UBM)** inference optimizations for **RFdiffusion3 (RFD3)** ([anthropics/uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling)).

Included model suites:
- **RFdiffusion3 (RFD3) with UBM Optimization**:
  - **CUDA Graphs (`RFD3_CUDAGRAPH=1`)**: Captures and replays the eager denoising step on GPU streams, removing Python loop overhead.
  - **Fused SwiGLU Transition (`RFD3_FZT=1`)**: Triton fused kernel served by `opt_core`.
  - **Triton Gather Attention (`RFD3_GATHER_ATTN=1`)**: Efficient gather kernel for pair-biased atom attention.
  - **PyTorch Inductor Compilation (`RFD3_COMPILE=1`)**: Ahead-of-time/JIT compilation of the four diffusion submodules (`encoder`, `diffusion_token_encoder`, `diffusion_transformer`, `decoder`) with dynamo cache boundaries.
  - **Initial Block Chunking & Hoist (`RFD3_INIT_CHUNK=1`, `RFD3_HOIST=1`)**: Dedup, downcast, and parallel pair features.
- **RFdiffusion3NA (RFD3NA)**: Nucleic acid and protein complex generative diffusion.
- **RoseTTAFold3 (RF3)**: All-atom structure prediction with implicit chirality and atom-level conditioning.
- **ProteinMPNN / LigandMPNN / SolubleMPNN**: Fast sequence design for fixed backbones and protein-ligand complexes.

All model weights (~7.9 GB across 6 checkpoints) are baked directly into the container under `/weights`.

---

### Usage

The container defaults to **`fast`** mode (`RFDIFFUSION3_OPT=fast`). You can also select **`exact`** (bit-identical designs to stock without torch.compile) or **`off`** (stock upstream logic).

#### Selecting Modes

Modes can be activated via:
1. **Environment variable** (works with standard `rfd3 design`):
   - `export RFDIFFUSION3_OPT=fast` (default in container)
   - `export RFDIFFUSION3_OPT=exact` (exact bit-matching numerics)
   - `export RFDIFFUSION3_OPT=off` (unoptimized stock)
2. **Dedicated CLI**:
   - `rfdiffusion3-opt design --mode fast inputs=... out_dir=...`
   - `rfdiffusion3-opt design --mode exact inputs=... out_dir=...`
   - `rfdiffusion3-opt check --mode fast` (validates levers without running)

#### Docker

```bash
# Interactive shell
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    nicholasfreitas/foundry:docker-0.2.0-UBM-RDF3

# Run RFD3 design (fast mode, default)
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    nicholasfreitas/foundry:docker-0.2.0-UBM-RDF3 \
    rfd3 design inputs=/workspace/test/rfd3_monomer.json out_dir=/workspace/output_rfd3

# Run RFD3 design in exact mode
docker run --gpus all --rm \
    -e RFDIFFUSION3_OPT=exact \
    -v $(pwd):/workspace \
    nicholasfreitas/foundry:docker-0.2.0-UBM-RDF3 \
    rfd3 design inputs=/workspace/test/rfd3_monomer.json out_dir=/workspace/output_rfd3_exact

# Run RF3 fold
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    nicholasfreitas/foundry:docker-0.2.0-UBM-RDF3 \
    rf3 fold inputs=/workspace/test/5vht_single.json out_dir=/workspace/output_rf3

# Run MPNN sequence design
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    nicholasfreitas/foundry:docker-0.2.0-UBM-RDF3 \
    mpnn --model_type protein_mpnn --structure_path /workspace/input.cif --out_directory /workspace/mpnn_out
```

#### Apptainer / Singularity

```bash
# Interactive shell with GPU
apptainer shell --nv foundry_0.2.0-UBM-RDF3.sif

# Run RFD3 design (fast mode)
apptainer run --nv foundry_0.2.0-UBM-RDF3.sif \
    rfd3 design inputs=test/rfd3_monomer.json out_dir=output_rfd3

# Run RFD3 design in exact mode
apptainer run --nv --env RFDIFFUSION3_OPT=exact foundry_0.2.0-UBM-RDF3.sif \
    rfd3 design inputs=test/rfd3_monomer.json out_dir=output_rfd3_exact

# Fully isolated offline run (no network, contained environment)
apptainer run --nv --containall --net --network=none \
    -B $(pwd)/output:/tmp/out \
    foundry_0.2.0-UBM-RDF3.sif \
    rfd3 design inputs=test/rfd3_monomer.json out_dir=/tmp/out

# Run RF3 fold
apptainer run --nv foundry_0.2.0-UBM-RDF3.sif \
    rf3 fold inputs=test/5vht_single.json out_dir=output_rf3

# Run MPNN sequence design
apptainer run --nv foundry_0.2.0-UBM-RDF3.sif \
    mpnn --model_type protein_mpnn --structure_path input.cif --out_directory mpnn_out
```

---

### Implementation Notes

1. **UBM Speed Optimization Integration**:
   - Anthropic's UBM shared kernel core (`opt_core` 0.5.228.0) and RFD3 acceleration driver (`rfdiffusion3_opt` 0.6.2) are installed into `/app/foundry/.venv`.
   - The 10-file RFD3 model overlay from `rfdiffusion3/opt/forward/xattempt_addon/patched/rfd3/model` replaces the corresponding files under `/app/foundry/.venv/lib/python3.12/site-packages/rfd3/model/`:
     - `RFD3_diffusion_module.py`
     - `cudagraph_sampler.py`
     - `hoist.py`
     - `inference_sampler.py`
     - `layers/attention.py`
     - `layers/block_utils.py`
     - `layers/blocks.py`
     - `layers/encoders.py`
     - `layers/layer_utils.py`
     - `layers/pairformer_layers.py`
   - An overlay marker `.xattempt_addon_overlay.json` is generated with byte-level SHA256 checksums ensuring complete overlay classification.
2. **Foundry 0.2.0 Compatibility Adjustments**:
   - `_COMPILE_TARGETS`: Added `_COMPILE_TARGETS = ("encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder")` to `rfd3.engine.RFD3InferenceEngine` to support torch.compile wrapping in `fast` mode.
   - `routes.py`: Upstream Foundry 0.2.0 uses PyTorch's native `nn.RMSNorm` rather than NVIDIA apex. Adjusted route expectation to accept PyTorch's native fallback RMSNorm when apex is absent.
3. **Baked Weights**:
   - `/weights/rf3_foundry_01_24_latest_remapped.ckpt` (2.83 GB)
   - `/weights/rfd3_latest.ckpt` (2.51 GB)
   - `/weights/rfd3na_1190.ckpt` (2.51 GB)
   - `/weights/proteinmpnn_v_48_020.pt` (6.7 MB)
   - `/weights/ligandmpnn_v_32_010_25.pt` (10.5 MB)
   - `/weights/solublempnn_v_48_020.pt` (6.7 MB)
4. **Apptainer Read-Only Filesystem & Cache Redirection**:
   - Triton, HuggingFace, Inductor, and Matplotlib caches are mapped to `/tmp` (`TRITON_CACHE_DIR=/tmp/triton`, `TORCHINDUCTOR_CACHE_DIR=/tmp/inductor`, `HF_HOME=/tmp/hf`, `XDG_CACHE_HOME=/tmp/cache`, `MPLCONFIGDIR=/tmp/mpl`).
   - PyTorch CUDA memory allocator is set to `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`.
   - `CC=/usr/bin/gcc` is exported for Triton JIT runtime compilation.

---

### To Do

- [ ] Benchmark complex multi-chain binder design workloads across varying diffusion batch sizes on cluster queues.
- [ ] Profile multi-GPU distributed design runs with Slurm orchestration.
