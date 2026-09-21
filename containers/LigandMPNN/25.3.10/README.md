# LigandMPNN Docker/Apptainer

Repo commit `26ec57ac` (2025-02-06). All 15 checkpoints baked at
`/workspace/LigandMPNN/model_params` — no download, no network.
SIF: `~/.ribbon/ribbon_containers/ligandMPNN_25.3.10.sif` (9.8 GB).

### Usage

One structure. `exec` with absolute paths is the robust form — it skips the
runscript, so nothing is written into the working directory:

```bash
apptainer exec --nv ligandMPNN_25.3.10.sif \
    python /workspace/LigandMPNN/run.py \
    --model_type ligand_mpnn \
    --checkpoint_ligand_mpnn /workspace/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
    --pdb_path input.pdb --out_folder ./out/ --batch_size 8 --seed 1
```

`run` also works and is shorter, because the runscript symlinks `LigandMPNN`
and `model_params` into the current directory so the relative default paths
resolve:

```bash
apptainer run --nv ligandMPNN_25.3.10.sif \
    python LigandMPNN/run.py --pdb_path input.pdb --out_folder ./out/
```

Both verified. The `run` form needs a writable working directory that does not
already contain `LigandMPNN` or `model_params` entries — the runscript does
`ln -s` under `set -euo pipefail`, so a name clash aborts before anything runs.
`exec` has neither constraint.

Docker: `docker run --gpus all -v $PWD:/data nicholasfreitas/ligandmpnn:docker-25.3.10 python /workspace/LigandMPNN/run.py --pdb_path /data/input.pdb --out_folder /data/out/`

### Many structures — `ligandmpnn_batch.py`

One `run.py` invocation costs **6.95 s** for a small monomer, of which **6.4 s
is Python startup, the torch import and CUDA context creation**. The checkpoint
itself is only 10 MB and `torch.load` accounts for 0.03 s — so the thing worth
amortising is process startup, not the weights.

`ligandmpnn_batch.py` (in this directory) designs a matrix of structures x
temperatures x seeds in one process. It calls upstream's own `run.main()` per
item and reuses upstream's argparse definition, so no design logic is
reimplemented and new upstream flags keep working via `--extra`.

```bash
apptainer exec --nv ligandMPNN_25.3.10.sif python ligandmpnn_batch.py \
    backbones/ --out run1 \
    --temperatures 0.1 0.2 --seeds 1 2 3 \
    --batch-size 8 --number-of-batches 2
```

**Measured: 40 structures in 39 s** (0.7 s each in steady state, 320 sequences)
against 278 s for 40 separate invocations — 7x at 40 structures, approaching 10x
as the run gets longer. Extrapolated: ~1000 backbones in ~12 min rather than
~1.9 h.

Upstream already has a `--pdb_path_multi` mode that loads the model once, so the
speed alone is not the reason to use this. The reason is that **upstream has no
per-structure error handling** — one unparseable PDB raises out of the loop and
abandons every structure after it. Verified: a 3-structure `--pdb_path_multi`
run with a bad file second raised `AttributeError: 'NoneType' object has no
attribute 'select'` out of `parse_PDB` and left 1 of 3 FASTAs written. The
driver adds:

- per-item isolation — a bad structure is logged to `logs/<uid>.log` and the run continues
- resume — `manifest.jsonl` is flushed per item, so re-invoking skips completed work and retries failures
- one aggregated `designs.csv` with sequence, `overall_confidence`, `ligand_confidence`, `seq_rec` per design, plus `designs.fasta`
- OOM retry at `batch_size 1` before giving an item up
- basename collision handling — `run.py` names outputs after the input basename
  (`run.py:368`), so `a/model.pdb` and `b/model.pdb` would silently overwrite;
  colliders are symlinked under distinct names
- `run.json` provenance: SIF path, repo commit, checkpoint sha256, driver sha256

Inputs: directories, globs, explicit paths, `--pdb-list`, or an upstream
`--pdb-multi` JSON. Models: `--model-type
{ligand_mpnn,protein_mpnn,soluble_mpnn,per_residue_label_membrane_mpnn,global_label_membrane_mpnn}`
with `--noise {002,005,010,020,030}` selecting the checkpoint (unavailable
combinations fail immediately with the list of what is baked in).
`--fixed-residues "A45 A46"` or per-structure `--fixed-residues-json`.
Anything else upstream accepts goes through verbatim, repeatable:
`--extra '--omit_AA C' --extra '--symmetry_residues "A12,A13"'`.
`--dry-run` prints the matrix and flags collisions without loading the model.

Sequences per structure = `--batch-size x --number-of-batches`. Prefer a larger
batch over more items: 8 sequences cost 0.87 s against 0.53 s for 1.

It is **not baked into the image** — same reasoning as the ESMFold2 container.
The image is the reproducible part; the driver is the part that keeps changing.

### Implementation Notes
- In general, I try to avoid using conda inside containers for simplicity. However, in this case I can't get PRODY to be installed properly without it. Conda it is!
- Avoiding conda makes the apptainer def significantly simpler.
- `run.py` picks the device itself (`cuda` if visible, else CPU) — there is no
  `--device` flag. Forgetting `--nv` silently runs on CPU; the driver prints
  which device it got.
- VRAM is not a constraint at this scale: 0.19-0.25 GB per item for a ~100-residue
  monomer, 6 GB for the 16-chain 2GFB example.

### To Do:
- Deleting symlinks still doesn't work... (`apptainer exec` avoids the runscript
  entirely, which sidesteps it — the batch driver uses `exec`.)
- 25.6.9 was never built: no SIF in `~/.ribbon/ribbon_containers` and no
  `nicholasfreitas/ligandmpnn:docker-25.6.9` in the local Docker daemon. 25.3.10
  is the working image and the one everything above was measured against. Note
  25.6.9 restructures the image (`/opt/conda/envs/app` via micromamba, versus
  25.3.10's `/usr/envs/ligandmpnn_env`), so `ligandmpnn_batch.py` will need
  `LIGANDMPNN_REPO` checked against the new layout when it is built.
