# ESMFold2 Docker/Apptainer

Biohub ESMFold2 and ESMFold2-Fast (Candido et al. 2026), all-atom co-folding of
protein + DNA/RNA/ligand. Fully stand-alone: **all weights are inside the
image**, including the 6B-parameter ESMC language model, and nothing is
downloaded or written outside `/tmp` at run time.

Version `26.9.5` is a date, not a software version: ESMFold2 has no release
number. The two Biohub forks it needs have no PyPI release either, so they are
pinned by commit in the Dockerfile.

### Usage
- Docker: `docker run -it --gpus all --shm-size=2g -v .:/workspace nicholasfreitas/esmfold2:docker-26.9.5 esmfold2-fold --input /app/test/1stp.json --out .`
- Apptainer: `apptainer run --nv esmfold2_26.9.5.sif esmfold2-fold --input /app/test/1stp.json --out .`

Bundled examples in `/app/test`: `1ubq.json` (ubiquitin, 76 aa, apo) and
`1stp.json` (streptavidin, 159 aa, biotin as CCD code `BTN`).

Sequence straight from the command line, ligand optional:

```bash
apptainer run --nv esmfold2_26.9.5.sif esmfold2-fold \
    --sequence MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG \
    --out out/ --num-diffusion-samples 1
```

```bash
# ligand by CCD code, or by SMILES for anything without a CCD entry
apptainer run --nv esmfold2_26.9.5.sif esmfold2-fold --sequence "$SEQ" --ligand-ccd BTN --out out/
apptainer run --nv esmfold2_26.9.5.sif esmfold2-fold --sequence "$SEQ" --ligand-smiles 'Cc1cc(=O)oc2c1ccc1nn[nH]c12' --out out/
```

`--model fast` selects ESMFold2-Fast (distilled trunk, no MSA support).
Output is one mmCIF per diffusion sample plus a JSON with per-sample pLDDT,
pTM, ipTM, timings and peak VRAM. `--help` lists the rest.

For anything beyond one chain plus one ligand — MSAs, multimers, covalent
bonds, pocket conditioning — import the builder directly; `esmfold2_fold.py`
is a thin convenience wrapper, not the full API:

```bash
apptainer exec --nv esmfold2_26.9.5.sif python -c "
import esmfold2_patches; esmfold2_patches.apply_patches()
from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
..."
```

### Batch runs — `esmfold2_batch.py`

`esmfold2-fold` loads the weights, folds once, and exits. Measured here: 65 s
load + 7 s Triton warm-up against **2.5 s** of actual folding, so a matrix of
folds run one process each spends ~96% of its wall time loading the same
12.7 GB.

`esmfold2_batch.py` (in this directory) loads once and folds a whole
sequences x ligand-conditions x seeds matrix. It is **deliberately not baked
into the image** — the container is the reproducible part (weights, pinned
forks, CCD), the driver is the part that keeps changing, and editing it must
not cost a rebuild. It needs no `--bind`: the image exports `PYTHONPATH=/app`,
so it imports the baked `esmfold2_patches` and weights from any host
directory, and Apptainer auto-binds `$HOME` and `$PWD`.

```bash
apptainer run --nv esmfold2_26.9.5.sif python esmfold2_batch.py \
    --fasta designs.fasta \
    --apo \
    --ligand TSA=smiles:'Cc1cc(=O)oc2c1ccc1nn[nH]c12' \
    --ligand BTN=ccd:BTN \
    --seeds 5 --num-diffusion-samples 5 \
    --out run1
```

Sequences from `--fasta`, repeatable `--sequence NAME=SEQ`, or `--config
run.json` (`{sequences, ligands, conditions, seeds, settings}`). A `:` in a
FASTA record splits protein chains. Conditions are repeatable `--ligand
NAME=ccd:CODE` (`+` joins cofactors) or `NAME=smiles:...`, plus `--apo`;
`--dry-run` prints the matrix and a VRAM ceiling without loading the model.

Per run it writes `<name>__<cond>__s<seed>__m<k>.cif`, a `manifest.jsonl`
flushed after every fold, a flat `summary.csv`, and a `run.json` recording the
SIF path, the driver's own sha256 and the baked weight revisions. Folds already
in the manifest are skipped on re-invocation, so an interrupted run resumes and
a crash costs one fold; failed folds are retried. Items run longest-first so a
size problem surfaces in the first minute rather than hour two, one bad item is
recorded and stepped over rather than killing the batch, and
`torch.cuda.empty_cache()` runs between folds — the caching allocator otherwise
holds its high-water mark, which is how a mixed-length batch OOMs on
fragmentation rather than on true demand.

Measured: 12 folds (2 sequences x 3 conditions x 2 seeds) in 1.6 min including
one load; steady state 2.5–4.0 s per fold.

**`esmfold2-fold --num-diffusion-samples 1` is broken.** At one sample
`builder.fold` returns a bare `MolecularComplexResult`, not a length-1 list, so
anything iterating the result raises `TypeError: 'MolecularComplexResult' object
is not iterable`. That hits the baked `esmfold2_fold.py`, whose own docstring
suggests 1 sample for a quick look. `esmfold2_batch.py` normalises the return
shape; against the baked CLI use `--num-diffusion-samples 2`. Fixing the baked
script needs a rebuild and a version bump — exactly the coupling that keeping
the driver outside the image avoids.

### Implementation Notes

**Weights (13.97 GB of the image).** Three repos are baked into `/app/models`:
`biohub/ESMC-6B`, `biohub/ESMFold2`, `biohub/ESMFold2-Fast`, with provenance
(repo, revision sha, source and stored byte counts) in
`/app/models/WEIGHTS.json`.

- **ESMC-6B is not optional.** Both trunk configs carry
  `esmc_id: "biohub/ESMC-6B"` and `ESMFold2Model.from_pretrained()` calls
  `load_esmc(config.esmc_id)` on the way out — without it neither checkpoint
  loads. One copy serves both trunks (`lm_num_layers: 80`, `lm_d_model: 2560`).
- **All three are stored bf16, not fp32 as published.** This is the one
  deliberate departure from the upstream bytes. `load_esmc`'s signature is
  `precision: str = "bf16"` and the trunks are loaded with
  `torch_dtype=torch.bfloat16`, so every one of these checkpoints is cast to
  bf16 at load in any ordinary run — storing them bf16 is bit-identical at
  inference and halves the weights:

  | repo | revision | published | stored |
  |---|---|---|---|
  | `biohub/ESMC-6B` | `45b0fa5d` | 25.41 GB | 12.70 GB |
  | `biohub/ESMFold2` | `8fc3ff47` | 1.36 GB | 0.89 GB |
  | `biohub/ESMFold2-Fast` | `c6c7958d` | 0.76 GB | 0.38 GB |

  (Trunk totals include `ccd.pkl` and the repo images, which are copied
  byte-for-byte; only tensors are cast — 8 shards across the three repos.)
  It does mean this image **cannot** run any of them in true fp32.
  `download_weights.py` does the cast shard-by-shard, deleting each fp32 shard
  as it goes, so peak build disk is ~19 GB rather than ~37 GB.
- Their `config.json` `dtype` field still reads `"float32"`, which is how
  Biohub ships it, and is left alone on purpose — that combination (bf16
  tensors, fp32 config field, `torch_dtype` passed explicitly at load) is what
  our validated runs used.
- `esmc_id` in both trunk configs is rewritten to `/app/models/ESMC-6B` at
  build time. Without that rewrite `from_pretrained` resolves ESMC through the
  HuggingFace cache, and a cache lookup against a read-only image is the
  failure mode that stopped weights being baked into the ESM3 container.
- **`ESMCFOLD_CCD_PATH` is load-bearing, and baking `ccd.pkl` is not enough.**
  `ESMFold2InputBuilder()` calls `load_ccd()`, which ignores `ESMFOLD2_MODELS`
  and `hf_hub_download`s `biohub/ESMFold2 ccd.pkl` into the HF cache unless
  this variable names an existing file. It is set to
  `/app/models/ESMFold2/ccd.pkl` in both the Dockerfile and `%environment`;
  note the spelling — `ESMCFOLD`, not `ESMFOLD2` — and that
  `conformers.py` reads it at *import* time. Without it the container folds
  only where a populated HF cache happens to be mounted, so it looks
  stand-alone on a machine that has one and fails everywhere else.
  `%test` asserts the variable resolves.
- Unlike ESM3, **these repos are ungated** — no HF token, at build or at run
  time. That is the whole reason this container can be stand-alone.

**Read-only filesystem.** `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are
set, and every cache that Torch, Triton, Inductor, HuggingFace or matplotlib
might write is redirected to `/tmp`, which Apptainer mounts writable even under
`--containall`. Verified by folding with `--net --network=none` and no host HF
cache bind-mounted; see `container_validation.json`.

**Three things are load-bearing at run time**, all handled inside
`esmfold2_fold.py` so a naive invocation cannot trip them:

1. **bf16 + `torch.autocast`.** fp32 inference needs ~26 GB and will not fit a
   16 GB card.
2. **`model.set_kernel_backend("fused")`.** The default (`None`) is roughly
   12x slower. The fused path is Triton, which JIT-builds C launcher stubs on
   the first fold — hence `build-essential` in the runtime stage and
   `TRITON_CACHE_DIR` in `/tmp`.
3. **A small-matrix SVD fallback** (`esmfold2_patches.apply_patches()`). The
   structure module takes SVDs of 3x3 frame matrices; under bf16 autocast those
   occasionally arrive non-finite and cuSOLVER raises instead of returning a
   factorisation. The patch routes 4x4-and-smaller SVDs to CPU fp32 with a NaN
   guard, which costs microseconds at that size. Apply it **before** importing
   the esm modules.

**Environment.** micromamba env `app` on **python 3.12**, not the template's
3.11: the esm fork declares `Requires-Python >=3.12,<3.13`. torch is
`2.7.1+cu126`, installed from the PyTorch cu126 index *before* the forks so
they resolve against it. The build gate asserts that exact version string, both
fork commit SHAs (read from each package's `direct_url.json`) and the two
`esmfold2` module paths on disk, so a quietly-swapped wheel or a fork that
drifted off its pin fails the build rather than the first fold. It deliberately
imports nothing: `import esm` initialises Triton's driver, which raises
`0 active drivers` without a GPU, and `docker build` has no GPU — the imports
and a real fold are checked in the GPU validation run instead. Note also that
the fork declares its `transformers` dependency as a direct URL on `@main`, so
installing both pins in one `pip` call raises `ResolutionImpossible`; `esm` goes
in first, then pinned `transformers` with `--no-deps --force-reinstall`.
`main` has already moved past `3a8956fb`, so that pin is what keeps this image
reproducing our earlier runs. Base image is
`cuda12.2.0-base-ubuntu22.04-micromamba` — the 12.6.0
variant has a directory in this repo but was never pushed to DockerHub, and it
would make no difference: the torch wheels carry their own CUDA 12.6 runtime
and only the host driver matters (tested on 535.309.01).

**Warm-up.** The first fold in a process pays ~15 s of Triton JIT.
`esmfold2_fold.py` does a small throwaway fold first so reported times are
steady state; `--no-warmup` skips it.

**Measured** on an RTX 4080 (16 GB), driver 535.309.01 — see
`container_validation.json` for the full record:

Folded under `--nv --containall --net --network=none` — no host `$HOME`, no host
HF cache, no route to the Hub — so these numbers come from the baked weights
alone. Full trunk, bf16, fused kernels, 5 diffusion samples, RMSD/TM after
sequence-alignment-based residue matching:

| target | tokens | fold | peak VRAM | pLDDT | CA RMSD (med) | TM (med) | ligand RMSD |
|---|---|---|---|---|---|---|---|
| 1UBQ (76 aa, apo) | 76 | 4.4 s | 13.98 GB | 0.81 | 0.94 Å | 0.951 | — |
| 1STP (159 aa + BTN) | 175 | 4.5 s | 15.75 GB | 0.86 | 0.38 Å | 0.991 | 0.48 Å |

CA RMSD ranges 0.64–1.35 Å over the five 1UBQ samples and 0.36–0.42 Å over the
five 1STP samples; biotin heavy-atom RMSD (16 atoms, matched by name, protein
CA superposition) ranges 0.44–0.91 Å, so ligand placement and not merely the
fold is reproduced. 1STP is compared over the 121 residues ordered in the
experimental entry, of 159 in SEQRES. ipTM averages 0.969 for the
protein–ligand interface.

**VRAM headroom is the practical limit.** 13.98 GB at 76 tokens and 15.75 GB at
175 tokens on a 16.0 GB card — ~12.7 GB of that is resident ESMC-6B, so
activations, not weights, are what run you out. With `chunk_size None` the full
trunk leaves roughly 0.6 GB at 175 tokens; past ~200 tokens on a 16 GB card
expect OOM and either set a chunk size or switch to `--model fast`. Larger
cards are unaffected.

Build wall time on this host: 333 s for the Docker rebuild with the builder
stage cached (the cold build including the weight bake is much longer), 2008 s
for `apptainer build`, the bulk of which is squashfs compression.

### To Do:
- MSA input is supported by the full trunk (`ProteinInput(msa=...)`) but nothing
  in the container generates alignments — no MMseqs2, no databases. Either add a
  search tool + reference DB (large) or keep passing precomputed MSAs in.
- fp8 ESMC (`precision="fp8"`) needs H100 + TransformerEngine ≥ 2.x, neither of
  which is here. Untested.
- Weights are baked, so the image is ~10x the size of a typical container in
  this repo. If that becomes a problem the alternative is a weights volume plus
  `ESMFOLD2_MODELS`, at the cost of the stand-alone property.
- ESMFold2-Fast is bundled but was not benchmarked in this build's validation.
