# ESMFold2-UBM Docker/Apptainer

Biohub ESMFold2 and ESMFold2-Fast (Candido et al. 2026), all-atom co-folding of
protein + DNA/RNA/ligand, updated with the speed and correctness optimization
patches from Anthropic's open-source
[uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling)
repository.

Fully stand-alone: **all weights are inside the image**, including the
6B-parameter ESMC language model, and nothing is downloaded or written outside
`/tmp` at run time.

Version `26.9.5-UBM` denotes the `26.9.5` base model release with the Anthropic
Uplifting Biomolecular Modeling (UBM) patches applied.

### Usage
- Docker: `docker run -it --gpus all --shm-size=2g -v .:/workspace nicholasfreitas/esmfold2:docker-26.9.5-UBM esmfold2-fold --input /app/test/1stp.json --out .`
- Apptainer: `apptainer run --nv esmfold2_26.9.5-UBM.sif esmfold2-fold --input /app/test/1stp.json --out .`

Bundled examples in `/app/test`: `1ubq.json` (ubiquitin, 76 aa, apo) and
`1stp.json` (streptavidin, 159 aa, biotin as CCD code `BTN`).

Sequence straight from the command line, ligand optional:

```bash
apptainer run --nv esmfold2_26.9.5-UBM.sif esmfold2-fold \
    --sequence MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG \
    --out out/ --num-diffusion-samples 1
```

```bash
# ligand by CCD code, or by SMILES for anything without a CCD entry
apptainer run --nv esmfold2_26.9.5-UBM.sif esmfold2-fold --sequence "$SEQ" --ligand-ccd BTN --out out/
apptainer run --nv esmfold2_26.9.5-UBM.sif esmfold2-fold --sequence "$SEQ" --ligand-smiles 'Cc1cc(=O)oc2c1ccc1nn[nH]c12' --out out/
```

`--model fast` selects ESMFold2-Fast (distilled trunk, no MSA support).
Output is one mmCIF per diffusion sample plus a JSON with per-sample pLDDT,
pTM, ipTM, timings and peak VRAM. `--help` lists the rest.

### UBM Optimization Patches Applied

The following patches from `anthropics/uplifting-biomolecular-modeling` (`opt/forward/fast_inference/upstream/`) are applied directly to the bundled codebase:

1. **U1 (`U1_swa_sdpa_mask_memoize.diff`)**: Memoize the SDPA sliding-window mask in `SWA3DRoPEAttention` (`modeling_esmfold2_common.py`). Eliminates re-computing identical boolean window masks for every SWA block and diffusion step (~15% faster sampler wall time in SDPA path).
2. **U2 (`U2_sampler_schedule_host_syncs.diff`)**: Eliminates 3 per-step `.item()` device-to-host synchronizations in the EDM sampler loop (`modeling_esmfold2_common.py`), converting schedule and gammas to Python floats once ahead of the loop.
3. **U3 (`U3_confidence_chain_pair_host_syncs.diff`)**: Eliminates redundant per-chain host sync in `ConfidenceHead.pair_chains_iptm` (`modeling_esmfold2.py`).
4. **U4 (`U4_trimul_stage3_row_pitch_alignment.diff`)**: Aligns stage-3 einsum operands to 16-byte row pitches (multiple of 8 elements) in `trimul_with_residual.py`. Allows cuBLAS to dispatch its fast sm90 batched GEMM kernel instead of slower unaligned fallbacks when sequence length `L % 8 != 0` (1.2-1.3x speedup on TriMul).
5. **U5 (`U5_pair_bias_int64_offsets.diff`)**: Promotes 32-bit row offsets to 64-bit (`int64`) in `fused_attention_pair_bias.py`. Eliminates integer overflow and illegal memory access crashes on long sequences / multiple diffusion samples.
6. **U6 (`U6_msa_features_vectorised.diff`)**: Vectorizes `msa_to_res_type_and_deletions` in `paired_msa.py` over an entire ASCII byte buffer instead of a character-by-character Python double loop (18-23x faster MSA parsing).

In addition, `esmfold2-fold` has been updated with return-type normalization so `--num-diffusion-samples 1` works without raising `TypeError`.

### Implementation Notes

**Weights (13.97 GB of the image).** Three repos are baked into `/app/models`:
`biohub/ESMC-6B`, `biohub/ESMFold2`, `biohub/ESMFold2-Fast`, with provenance
(repo, revision sha, source and stored byte counts) in
`/app/models/WEIGHTS.json`.

- **All three are stored bf16, not fp32 as published.** `load_esmc`'s signature is
  `precision: str = "bf16"` and the trunks are loaded with
  `torch_dtype=torch.bfloat16`, so every one of these checkpoints is cast to
  bf16 at load in any ordinary run — storing them bf16 is bit-identical at
  inference and halves the weights:

  | repo | revision | published | stored |
  |---|---|---|---|
  | `biohub/ESMC-6B` | `45b0fa5d` | 25.41 GB | 12.70 GB |
  | `biohub/ESMFold2` | `8fc3ff47` | 1.36 GB | 0.89 GB |
  | `biohub/ESMFold2-Fast` | `c6c7958d` | 0.76 GB | 0.38 GB |

- `esmc_id` in both trunk configs is rewritten to `/app/models/ESMC-6B` at
  build time.
- `ESMCFOLD_CCD_PATH` is set to `/app/models/ESMFold2/ccd.pkl` in both Dockerfile
  and `%environment`.
- `flash-attn 2.8.3.post1` is included for accelerated attention on Ampere,
  Ada Lovelace (RTX 40-series), and Hopper cards.

**Read-only filesystem.** `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are
set, and every cache that Torch, Triton, Inductor, HuggingFace or matplotlib
might write is redirected to `/tmp`, which Apptainer mounts writable even under
`--containall`.

### Validation & Benchmarks

Validated on host with NVIDIA GeForce RTX 4080 (16 GB VRAM), driver `535.309.01`, under complete isolation (`apptainer run --nv --containall --net --network=none`):

| target | tokens | fold | peak VRAM | pLDDT (mean) | CA RMSD (med) | TM-score (med) | ligand RMSD (med) |
|---|---|---|---|---|---|---|---|
| 1UBQ (76 aa, apo) | 76 | 4.59 s | 13.86 GB | 0.801 | 0.91 Å | 0.955 | — |
| 1STP (159 aa + BTN) | 175 | 4.77 s | 15.60 GB | 0.882 | 0.31 Å | 0.994 | 0.89 Å |

- **CA RMSD**: 0.64–1.30 Å across 1UBQ samples; 0.30–0.34 Å across 1STP samples.
- **Biotin (BTN) heavy-atom RMSD**: 0.46–0.90 Å over 16 ligand atoms matched by name on protein CA frame.
- **Interface pTM (ipTM)**: 0.979 mean for Streptavidin–Biotin complex.
- **Build Wall Time**: Docker build: ~10.4 min (reusing baked weights); Apptainer build: 24.5 min (including squashfs compression).


### To Do:
- Add MMseqs2 support for in-container MSA generation for `full_msa` workflows.
- Explore fp8 ESMC inference for Hopper (H100/H200) architectures.
