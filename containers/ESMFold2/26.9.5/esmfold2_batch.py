#!/usr/bin/env python
"""Batch ESMFold2 co-folding: sequences x ligand conditions x seeds, one model load.

Runs INSIDE the ESMFold2 container, but lives outside it so it can be edited
without a rebuild. The image exports PYTHONPATH=/app, so the baked
esmfold2_patches and weights are importable from any host directory:

  apptainer run --nv esmfold2_26.9.5.sif python esmfold2_batch.py --help

Loading the 12.7 GB of weights costs ~90 s per process; a fold costs ~3 s.
Everything here exists to pay that 90 s once.

Inputs -- any one of:
  --fasta designs.fasta          ':' in a record splits protein chains (multimer)
  --sequence NAME=SEQUENCE       repeatable, for quick one-offs
  --config run.json             {sequences, ligands, conditions, seeds, settings}

Ligand conditions -- repeatable, applied to every sequence:
  --ligand TSA=smiles:'Cc1cc(=O)oc2c1ccc1nn[nH]c12'
  --ligand BTN=ccd:BTN           '+' joins co-factors: ccd:BTN+HEM
  --apo                          add a ligand-free condition
Given none, the run is apo-only.

Outputs, under --out:
  <name>__<cond>__s<seed>__m<k>.cif   one mmCIF per diffusion sample
  manifest.jsonl                      one row per finished fold, flushed as it goes
  summary.csv                         flat table, rebuilt from the manifest
  run.json                            resolved matrix + container provenance

Re-running skips folds already in the manifest, so a crash costs one fold.
Use --overwrite to force, --dry-run to print the matrix and VRAM estimate.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

import esmfold2_patches  # from /app inside the container

PATCHES = esmfold2_patches.apply_patches()

from esm.models.esmfold2 import (  # noqa: E402
    ESMFold2InputBuilder, LigandInput, ProteinInput, StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: E402

MODELS = Path(os.environ.get("ESMFOLD2_MODELS", "/app/models"))
CKPT = {"full": "ESMFold2", "fast": "ESMFold2-Fast"}
CHAINS = "ABCDEFGHIJKMNOPQRSTUVWXYZ"  # L reserved for ligand

# Peak VRAM ~= weights + k * tokens^2, fitted on this container at 5 diffusion
# samples: 13.98 GB at 76 tokens, 15.85 GB at 175 (full trunk, bf16,
# chunk_size=None). Activations also grow with num_diffusion_samples, so this
# OVER-estimates below 5 (measured 14.50 GB at 175 tokens / 2 samples).
# Advisory only: a conservative ceiling for "will this run fit".
VRAM_FLOOR_GB, VRAM_K = 13.47, (15.85 - 13.47) / 175 ** 2


def est_vram_gb(tokens):
    return round(VRAM_FLOOR_GB + VRAM_K * tokens ** 2, 2)


def parse_fasta(path):
    out, name, buf = [], None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if name:
                out.append((name, "".join(buf)))
            name, buf = line[1:].split()[0], []
        elif line:
            buf.append(line)
    if name:
        out.append((name, "".join(buf)))
    return out


def parse_ligand(spec):
    """'NAME=ccd:BTN+HEM' or 'NAME=smiles:CCO' -> (name, {ccd|smiles})."""
    if "=" not in spec:
        raise SystemExit(f"--ligand needs NAME=SPEC, got {spec!r}")
    name, body = spec.split("=", 1)
    return name, normalise_ligand(body)


def normalise_ligand(body):
    """Accept 'ccd:X', 'smiles:X', {'ccd':...}, {'smiles':...}, or a bare SMILES."""
    if isinstance(body, dict):
        if body.get("ccd"):
            ccd = body["ccd"]
            return {"ccd": ccd if isinstance(ccd, list) else [ccd]}
        return {"smiles": body["smiles"]}
    if body.startswith("ccd:"):
        return {"ccd": body[4:].split("+")}
    if body.startswith("smiles:"):
        return {"smiles": body[7:]}
    return {"smiles": body}  # bare string = SMILES; use ccd: for CCD codes


def build_matrix(a):
    seqs, ligands, conditions, seeds, settings = [], {}, [], None, {}

    if a.config:
        cfg = json.load(open(a.config))
        raw = cfg.get("sequences", {})
        if isinstance(raw, dict):
            seqs += list(raw.items())
        else:
            seqs += [(r["name"], r["sequence"]) for r in raw]
        ligands.update({k: normalise_ligand(v) for k, v in cfg.get("ligands", {}).items()})
        conditions += list(cfg.get("conditions", []))
        seeds = cfg.get("seeds", None)
        settings.update(cfg.get("settings", {}))

    if a.fasta:
        seqs += parse_fasta(a.fasta)
    for s in a.sequence or []:
        if "=" not in s:
            raise SystemExit(f"--sequence needs NAME=SEQUENCE, got {s!r}")
        n, q = s.split("=", 1)
        seqs.append((n, q))

    for spec in a.ligand or []:
        n, body = parse_ligand(spec)
        ligands[n] = body
        if n not in conditions:
            conditions.append(n)
    if a.apo and "apo" not in conditions:
        conditions.insert(0, "apo")
    if not conditions:
        conditions = ["apo"]
    if a.conditions:
        conditions = a.conditions

    # CLI overrides config
    if a.seeds is not None:
        seeds = a.seeds
    if seeds is None:
        seeds = [0]
    if isinstance(seeds, int):
        seeds = list(range(seeds))
    if a.seed_list:
        seeds = [int(x) for x in a.seed_list.split(",")]

    for k, v in (("num_loops", a.num_loops), ("num_sampling_steps", a.num_sampling_steps),
                 ("num_diffusion_samples", a.num_diffusion_samples), ("model", a.model),
                 ("chunk_size", a.chunk_size)):
        if v is not None:
            settings[k] = v
    settings.setdefault("num_loops", 10)
    settings.setdefault("num_sampling_steps", 68)
    settings.setdefault("num_diffusion_samples", 5)
    settings.setdefault("model", "full")
    settings.setdefault("chunk_size", None)

    if not seqs:
        raise SystemExit("no sequences: pass --fasta, --sequence or --config")
    unknown = [c for c in conditions if c != "apo" and c not in ligands]
    if unknown:
        raise SystemExit(f"conditions with no ligand defined: {unknown}")
    dupes = {n for n, _ in seqs if [x for x, _ in seqs].count(n) > 1}
    if dupes:
        raise SystemExit(f"duplicate sequence names: {sorted(dupes)}")
    bad = [n for n, _ in seqs if "__" in n]
    if bad:
        raise SystemExit(f"'__' is the output field separator; rename: {bad}")

    jobs = []
    for name, seq in seqs:
        chains = seq.split(":")
        n_res = sum(len(c) for c in chains)
        for cond in conditions:
            lig = None if cond == "apo" else ligands[cond]
            n_lig = len(lig.get("ccd", [])) * 12 if lig and lig.get("ccd") else (
                sum(1 for ch in lig["smiles"] if ch.isalpha()) if lig else 0)
            for sd in seeds:
                jobs.append(dict(name=name, chains=chains, n_res=n_res, cond=cond,
                                 ligand=lig, seed=sd, tokens_est=n_res + n_lig))
    # longest first: a size problem surfaces in the first minute, not hour two
    jobs.sort(key=lambda j: -j["tokens_est"])
    return jobs, ligands, conditions, seeds, settings


def fold_one(builder, model, job, settings):
    chains = [ProteinInput(id=CHAINS[i], sequence=s) for i, s in enumerate(job["chains"])]
    if job["ligand"]:
        chains.append(LigandInput(id="L", ccd=job["ligand"].get("ccd"),
                                  smiles=job["ligand"].get("smiles")))
    spi = StructurePredictionInput(sequences=chains)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = builder.fold(model, spi,
                           num_loops=settings["num_loops"],
                           num_sampling_steps=settings["num_sampling_steps"],
                           num_diffusion_samples=settings["num_diffusion_samples"],
                           seed=job["seed"])
    # At num_diffusion_samples=1 builder.fold returns a bare MolecularComplexResult,
    # not a length-1 list. Normalise so downstream code has one shape to handle.
    # (The container's own esmfold2_fold.py has this bug at --num-diffusion-samples 1.)
    return list(out) if isinstance(out, (list, tuple)) else [out]


def num(x):
    if x is None:
        return None
    t = torch.as_tensor(x).float()
    return round(float(t), 4) if t.numel() == 1 else [round(v, 4) for v in t.reshape(-1).tolist()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta")
    ap.add_argument("--sequence", action="append")
    ap.add_argument("--config")
    ap.add_argument("--ligand", action="append")
    ap.add_argument("--apo", action="store_true")
    ap.add_argument("--conditions", nargs="+", help="subset of the defined conditions to run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, help="N -> seeds 0..N-1")
    ap.add_argument("--seed-list", help="explicit, e.g. 0,7,42")
    ap.add_argument("--num-diffusion-samples", type=int)
    ap.add_argument("--num-loops", type=int)
    ap.add_argument("--num-sampling-steps", type=int)
    ap.add_argument("--model", choices=list(CKPT))
    ap.add_argument("--chunk-size", type=int, help="set for long targets; default None")
    ap.add_argument("--limit", type=int, help="run only the first N folds (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="re-run folds already in the manifest")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()

    jobs, ligands, conditions, seeds, settings = build_matrix(a)
    if a.limit:
        jobs = jobs[:a.limit]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"

    done = set()
    if manifest.exists() and not a.overwrite:
        for line in manifest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("ok"):
                    done.add((r["name"], r["cond"], r["seed"]))
    todo = [j for j in jobs if (j["name"], j["cond"], j["seed"]) not in done]

    biggest = max(jobs, key=lambda j: j["tokens_est"])
    print(f"MATRIX {len(jobs)} folds = {len({j['name'] for j in jobs})} seq "
          f"x {len(conditions)} cond {conditions} x {len(seeds)} seeds; "
          f"{len(done)} already done, {len(todo)} to run", flush=True)
    print(f"LARGEST ~{biggest['tokens_est']} tokens ({biggest['name']}/{biggest['cond']}), "
          f"est. peak <={est_vram_gb(biggest['tokens_est'])} GB "
          f"(ceiling at 5 samples; chunk_size={settings['chunk_size']})", flush=True)
    print(f"SETTINGS {json.dumps(settings)}", flush=True)

    run_rec = dict(
        settings=settings, conditions=conditions, seeds=seeds, ligands=ligands,
        n_folds=len(jobs), patches=PATCHES, torch=torch.__version__,
        container=os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"),
        container_name=os.environ.get("APPTAINER_NAME") or os.environ.get("SINGULARITY_NAME"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        models_dir=str(MODELS), argv=sys.argv[1:], started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    wpath = MODELS / "WEIGHTS.json"
    if wpath.exists():
        run_rec["weights"] = json.load(open(wpath)).get("models", {})
    (out / "run.json").write_text(json.dumps(run_rec, indent=1))

    if a.dry_run:
        for j in jobs[:20]:
            print(f"  {j['name']:<24} {j['cond']:<8} seed={j['seed']:<4} "
                  f"~{j['tokens_est']} tok  est {est_vram_gb(j['tokens_est'])} GB")
        if len(jobs) > 20:
            print(f"  ... {len(jobs) - 20} more")
        return 0
    if not todo:
        print("NOTHING TO DO (all folds present; --overwrite to redo)", flush=True)
        return 0

    t0 = time.time()
    model_dir = MODELS / CKPT[settings["model"]]
    model = ESMFold2Model.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda().eval()
    model.set_kernel_backend("fused")        # REQUIRED: default None is ~12x slower
    model.set_chunk_size(settings["chunk_size"])
    builder = ESMFold2InputBuilder(ccd_cache=MODELS / "ESMFold2")
    print(f"LOADED {CKPT[settings['model']]} in {time.time() - t0:.1f}s "
          f"({torch.cuda.memory_reserved() / 1e9:.2f} GB reserved)", flush=True)

    if not a.no_warmup:
        t = time.time()
        fold_one(builder, model, dict(chains=[todo[0]["chains"][0][:40]], cond="apo",
                                      ligand=None, seed=999),
                 dict(settings, num_loops=3, num_sampling_steps=14, num_diffusion_samples=1))
        print(f"WARMUP {time.time() - t:.1f}s", flush=True)

    fh = manifest.open("a")
    n_ok = n_fail = 0
    for i, j in enumerate(todo, 1):
        tag = f"{j['name']}__{j['cond']}__s{j['seed']}"
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            preds = fold_one(builder, model, j, settings)
            fold_s = round(time.time() - t, 2)
            cifs = []
            for k, p in enumerate(preds):
                fn = f"{tag}__m{k}.cif"
                (out / fn).write_text(p.complex.to_mmcif())
                cifs.append(fn)
            rec = dict(ok=True, name=j["name"], cond=j["cond"], seed=j["seed"],
                       n_res=j["n_res"], n_chains=len(j["chains"]), ligand=j["ligand"],
                       n_tokens=int(torch.as_tensor(preds[0].plddt).reshape(-1).shape[0]),
                       plddt=[num(p.plddt.float().mean() if hasattr(p.plddt, "float")
                                  else p.plddt) for p in preds],
                       ptm=[num(p.ptm) for p in preds],
                       # ipTM is inter-chain: meaningless for a single-chain apo
                       # fold, where the model returns 0.0. Blank it so it is not
                       # mistaken for a measured value.
                       iptm=([num(p.iptm) for p in preds]
                             if len(j["chains"]) + bool(j["ligand"]) > 1 else None),
                       fold_s=fold_s, peak_vram_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2),
                       checkpoint=CKPT[settings["model"]], cif=cifs)
            del preds
            n_ok += 1
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            rec = dict(ok=False, name=j["name"], cond=j["cond"], seed=j["seed"],
                       n_res=j["n_res"], error="cuda_oom", detail=str(e)[:200],
                       hint="retry with --chunk-size 128, or --model fast")
            n_fail += 1
        except Exception as e:                      # one bad item must not kill the batch
            rec = dict(ok=False, name=j["name"], cond=j["cond"], seed=j["seed"],
                       n_res=j["n_res"], error=type(e).__name__, detail=str(e)[:200])
            n_fail += 1
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        # the caching allocator holds the high-water mark across folds; release it
        # so a long item followed by a longer one does not OOM on fragmentation
        torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {tag} "
              + (f"{rec['fold_s']}s plddt={rec['plddt'][0]} peak={rec['peak_vram_gb']}GB"
                 if rec["ok"] else f"FAILED {rec['error']}")
              + f" | elapsed {el/60:.1f}m, eta {el/i*(len(todo)-i)/60:.1f}m", flush=True)
    fh.close()

    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    cols = ["ok", "name", "cond", "seed", "n_res", "n_tokens", "checkpoint", "seed_samples",
            "plddt_mean", "plddt_best", "ptm_mean", "iptm_mean", "fold_s", "peak_vram_gb",
            "error", "cif"]
    with (out / "summary.csv").open("w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            pl = [x for x in (r.get("plddt") or []) if x is not None]
            pt = [x for x in (r.get("ptm") or []) if x is not None]
            it = [x for x in (r.get("iptm") or []) if x is not None]
            w.writerow(dict(r, seed_samples=len(r.get("cif") or []),
                            plddt_mean=round(sum(pl) / len(pl), 4) if pl else None,
                            plddt_best=max(pl) if pl else None,
                            ptm_mean=round(sum(pt) / len(pt), 4) if pt else None,
                            iptm_mean=round(sum(it) / len(it), 4) if it else None,
                            cif=";".join(r.get("cif") or [])))
    print(f"DONE {n_ok} ok, {n_fail} failed, {(time.time() - t0)/60:.1f} min total "
          f"-> {out}/summary.csv", flush=True)
    return 1 if n_fail and not n_ok else 0


if __name__ == "__main__":
    sys.exit(main())
