"""Fold one protein (optionally with one ligand) with the baked ESMFold2 weights.

  esmfold2-fold --input /app/test/1stp.json --out out/
  esmfold2-fold --sequence MQIFVKTL... --out out/
  esmfold2-fold --sequence MQIF... --ligand-ccd BTN --out out/
  esmfold2-fold --sequence MQIF... --ligand-smiles 'Cc1cc(=O)oc2c1ccc1nn[nH]c12' --out out/

Defaults follow the FoldBench inference protocol in Candido et al. 2026:
num_loops=10, num_sampling_steps=68, num_diffusion_samples=5. Drop
--num-diffusion-samples to 1 for a quick look.

Writes, under --out:
  <name>__m<k>.cif    one mmCIF per diffusion sample
  <name>.json         per-sample pLDDT / pTM / ipTM, timings, peak VRAM, settings

Three things in here are load-bearing on a 16 GB card; see the container README
before editing:
  1. bf16 weights folded under torch.autocast - fp32 needs ~26 GB.
  2. the small-matrix SVD patch in esmfold2_patches.
  3. kernel_backend="fused" - the default (None) is ~12x slower.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

import esmfold2_patches

PATCHES = esmfold2_patches.apply_patches()

from esm.models.esmfold2 import (  # noqa: E402
    ESMFold2InputBuilder, LigandInput, ProteinInput, StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: E402

MODELS = os.environ.get("ESMFOLD2_MODELS", "/app/models")
CKPT = {"full": "ESMFold2", "fast": "ESMFold2-Fast"}


def num(x):
    if x is None:
        return None
    t = torch.as_tensor(x).float()
    return float(t) if t.numel() == 1 else t.reshape(-1).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence", help="protein sequence, one chain")
    src.add_argument("--input", help="JSON with {name, protein:{id,sequence}, ligand:{id,ccd|smiles}, seed}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default="full", choices=list(CKPT))
    lig = ap.add_mutually_exclusive_group()
    lig.add_argument("--ligand-ccd", nargs="+", help="one or more CCD codes, e.g. BTN")
    lig.add_argument("--ligand-smiles")
    ap.add_argument("--protein-chain", default="A")
    ap.add_argument("--ligand-chain", default="L")
    ap.add_argument("--num-loops", type=int, default=10)
    ap.add_argument("--num-sampling-steps", type=int, default=68)
    ap.add_argument("--num-diffusion-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the discarded warm-up fold (Triton JIT, ~15 s)")
    a = ap.parse_args()

    if a.input:
        spec = json.load(open(a.input))
        seq = spec["protein"]["sequence"]
        pchain = spec["protein"].get("id", a.protein_chain)
        ligspec = spec.get("ligand")
        name = a.name or spec.get("name") or os.path.basename(a.input).split(".")[0]
        seed = spec.get("seed", a.seed)
    else:
        seq, pchain, name, seed = a.sequence, a.protein_chain, a.name or "fold", a.seed
        ligspec = None
        if a.ligand_ccd:
            ligspec = dict(id=a.ligand_chain, ccd=list(a.ligand_ccd))
        elif a.ligand_smiles:
            ligspec = dict(id=a.ligand_chain, smiles=a.ligand_smiles)

    os.makedirs(a.out, exist_ok=True)
    model_dir = os.path.join(MODELS, CKPT[a.model])
    wpath = os.path.join(MODELS, "WEIGHTS.json")
    weights = json.load(open(wpath)) if os.path.exists(wpath) else {}

    t0 = time.time()
    model = ESMFold2Model.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda().eval()
    model.set_kernel_backend("fused")   # REQUIRED: default None is ~12x slower
    model.set_chunk_size(None)          # optimal and OOM-safe for L <= 1024
    load_s = round(time.time() - t0, 2)
    # ccd_cache is a *directory* holding ccd.pkl. ESMCFOLD_CCD_PATH (set in the
    # image) already wins over this, but pass it so the script also works
    # offline outside the container, where that var may be unset.
    builder = ESMFold2InputBuilder(
        ccd_cache=Path(os.environ.get("ESMFOLD2_MODELS", "/app/models")) / "ESMFold2")

    def fold(loops, steps, samples, sd):
        chains = [ProteinInput(id=pchain, sequence=seq)]
        if ligspec:
            chains.append(LigandInput(id=ligspec.get("id", a.ligand_chain),
                                      ccd=ligspec.get("ccd"),
                                      smiles=ligspec.get("smiles")))
        spi = StructurePredictionInput(sequences=chains)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return builder.fold(model, spi, num_loops=loops, num_sampling_steps=steps,
                                num_diffusion_samples=samples, seed=sd)

    # Triton kernels JIT-compile on the first call. Discard it so the reported
    # fold time is steady state, not warm-up.
    warmup_s = None
    if not a.no_warmup:
        t0 = time.time()
        fold(3, 14, 1, 999)
        warmup_s = round(time.time() - t0, 2)
        print(f"WARMUP {warmup_s}s", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    preds = fold(a.num_loops, a.num_sampling_steps, a.num_diffusion_samples, seed)
    fold_s = round(time.time() - t0, 2)

    for k, p in enumerate(preds):
        with open(os.path.join(a.out, f"{name}__m{k}.cif"), "w") as fh:
            fh.write(p.complex.to_mmcif())

    rec = dict(
        name=name, checkpoint=CKPT[a.model], model_dir=model_dir,
        n_residues=len(seq), ligand=ligspec, protein_chain=pchain, seed=seed,
        settings=dict(num_loops=a.num_loops, num_sampling_steps=a.num_sampling_steps,
                      num_diffusion_samples=a.num_diffusion_samples, dtype="bfloat16",
                      autocast="cuda/bfloat16", kernel_backend="fused", chunk_size=None),
        patches=PATCHES,
        plddt_mean=[round(float(torch.as_tensor(p.plddt).float().mean()), 4) for p in preds],
        ptm=[num(p.ptm) for p in preds],
        iptm=[num(p.iptm) for p in preds],
        n_tokens=int(torch.as_tensor(preds[0].plddt).reshape(-1).shape[0]),
        load_s=load_s, warmup_s=warmup_s, fold_s=fold_s,
        peak_vram_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2),
        torch=torch.__version__, weights=weights.get("models", {}).get(CKPT[a.model]),
        cif=[f"{name}__m{k}.cif" for k in range(len(preds))])
    with open(os.path.join(a.out, f"{name}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print("FOLD " + json.dumps({k: rec[k] for k in
          ("name", "checkpoint", "n_tokens", "fold_s", "peak_vram_gb",
           "plddt_mean", "iptm")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
