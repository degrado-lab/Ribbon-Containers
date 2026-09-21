#!/usr/bin/env python
"""Batch runner for LigandMPNN / ProteinMPNN / SolubleMPNN inside the
LigandMPNN container.

Designs a matrix of structures x temperatures x seeds in ONE process, so
Python startup + torch import + CUDA context (measured: 6.4 s of the 6.95 s
a one-structure `run.py` invocation costs) are paid once instead of per
structure. It calls upstream's own `run.main()` per item -- no design logic
is reimplemented here -- and adds what upstream's `--pdb_path_multi` mode
lacks: per-structure failure isolation (upstream aborts the whole run on one
unparseable PDB), resume, progress, and one aggregated sequence table at the
end.

  apptainer exec --nv ligandMPNN_25.3.10.sif \
      python ligandmpnn_batch.py backbones/ --out run1 \
             --temperatures 0.1 0.2 --seeds 1 2 --batch-size 8

Outputs under --out:
  seqs/<name>__T<t>__s<seed>.fa   upstream FASTA, one per item
  designs.csv                     every sequence, one row, with confidences
  designs.fasta                    all designed sequences, natives excluded
  manifest.jsonl                  one record per item, flushed as it goes
  run.json                        provenance
  logs/<uid>.log                  upstream stdout, kept for failures only
"""
import argparse
import csv
import glob
import hashlib
import io
import json
import os
import contextlib
import shlex
import subprocess
import sys
import textwrap
import time
import traceback

REPO = os.environ.get("LIGANDMPNN_REPO", "/workspace/LigandMPNN")
PARAMS = os.environ.get("LIGANDMPNN_PARAMS", os.path.join(REPO, "model_params"))

# model_type -> (checkpoint flag, filename glob with {noise})
MODELS = {
    "ligand_mpnn": ("--checkpoint_ligand_mpnn", "ligandmpnn_v_32_{noise}_25.pt"),
    "protein_mpnn": ("--checkpoint_protein_mpnn", "proteinmpnn_v_48_{noise}.pt"),
    "soluble_mpnn": ("--checkpoint_soluble_mpnn", "solublempnn_v_48_{noise}.pt"),
    "per_residue_label_membrane_mpnn": (
        "--checkpoint_per_residue_label_membrane_mpnn",
        "per_residue_label_membrane_mpnn_v_48_{noise}.pt"),
    "global_label_membrane_mpnn": (
        "--checkpoint_global_label_membrane_mpnn",
        "global_label_membrane_mpnn_v_48_{noise}.pt"),
}
SC_CHECKPOINT = "ligandmpnn_sc_v_32_002_16.pt"
DEFAULT_NOISE = {"ligand_mpnn": "010", "protein_mpnn": "020", "soluble_mpnn": "020",
                 "per_residue_label_membrane_mpnn": "020",
                 "global_label_membrane_mpnn": "020"}


def sha256(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
            if cap and fh.tell() > cap:
                break
    return h.hexdigest()


def upstream_parser():
    """Reuse run.py's own argparse definition.

    run.py builds its parser inside the `__main__` guard, so importing the
    module does not expose it. Exec the guard body with the trailing
    `parse_args()`/`main()` dispatch removed. This tracks upstream when they
    add or change arguments, which copying 50 add_argument calls in here
    would not.
    """
    src = open(os.path.join(REPO, "run.py")).read()
    guard = 'if __name__ == "__main__":'
    if guard not in src:
        sys.exit(f"cannot find the __main__ guard in {REPO}/run.py -- upstream "
                 "layout changed; driver needs updating")
    body = textwrap.dedent(src.split(guard, 1)[1]).split("args = argparser.parse_args()")[0]
    ns = {"argparse": argparse}
    exec(compile(body, "run.py:parser", "exec"), ns)
    p = ns["argparser"]
    if len(p._actions) < 40:
        sys.exit(f"only parsed {len(p._actions)} upstream arguments -- refusing "
                 "to run with a truncated parser")
    return p


def collect_structures(items, pdb_list, pdb_multi, exts):
    """Accept files, directories, globs, a list file, or an upstream multi JSON."""
    paths, per_path_opts = [], {}
    for it in items:
        if os.path.isdir(it):
            for e in exts:
                paths += sorted(glob.glob(os.path.join(it, f"*{e}")))
        elif any(ch in it for ch in "*?["):
            paths += sorted(glob.glob(it))
        else:
            paths.append(it)
    if pdb_list:
        with open(pdb_list) as fh:
            paths += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if pdb_multi:
        with open(pdb_multi) as fh:
            d = json.load(fh)
        paths += list(d)
        per_path_opts = {k: v for k, v in d.items() if v}
    seen, out = set(), []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    missing = [p for p in out if not os.path.isfile(p)]
    if missing:
        sys.exit(f"{len(missing)} input(s) do not exist, first: {missing[0]}")
    return out, per_path_opts


def stage_unique_names(paths, stage_dir):
    """run.py names every output after the input basename (run.py:368), so two
    inputs called model.pdb in different directories would overwrite each
    other. Only when that happens, symlink the colliders under distinct names
    and hand run.py the link."""
    stems = {}
    for p in paths:
        stems.setdefault(os.path.splitext(os.path.basename(p))[0], []).append(p)
    resolved = {}
    dupes = {s: ps for s, ps in stems.items() if len(ps) > 1}
    if dupes:
        os.makedirs(stage_dir, exist_ok=True)
    for stem, ps in stems.items():
        if len(ps) == 1:
            resolved[ps[0]] = (stem, ps[0])
            continue
        for p in ps:
            uniq = f"{stem}_{hashlib.sha1(p.encode()).hexdigest()[:6]}"
            link = os.path.join(stage_dir, uniq + os.path.splitext(p)[1])
            if not os.path.lexists(link):
                os.symlink(p, link)
            resolved[p] = (uniq, link)
    return resolved, dupes


def parse_fasta(path):
    """Upstream headers are comma-separated key=value after the name. The
    native input sequence is written first and carries no id= field."""
    recs, name, hdr, seq = [], None, None, []
    with open(path) as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln.startswith(">"):
                if hdr is not None:
                    recs.append((name, hdr, "".join(seq)))
                parts = [x.strip() for x in ln[1:].split(",")]
                name, seq = parts[0], []
                hdr = {}
                for kv in parts[1:]:
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        hdr[k.strip()] = v.strip()
            else:
                seq.append(ln.strip())
    if hdr is not None:
        recs.append((name, hdr, "".join(seq)))
    return recs


def main():
    ap = argparse.ArgumentParser(
        description="Batch LigandMPNN over many structures with one model load.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("structures", nargs="*", help="PDB/CIF files, directories, or globs")
    ap.add_argument("--pdb-list", help="text file, one structure path per line")
    ap.add_argument("--pdb-multi", help="upstream --pdb_path_multi JSON")
    ap.add_argument("--ext", nargs="+", default=[".pdb", ".cif"],
                    help="extensions picked up when a directory is given")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--model-type", default="ligand_mpnn", choices=sorted(MODELS))
    ap.add_argument("--noise", help="checkpoint noise level, e.g. 005/010/020/030 "
                                    "(default depends on model type)")
    ap.add_argument("--checkpoint", help="explicit checkpoint path, overrides --model-type/--noise")
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1],
                    help="0 lets upstream draw a random seed")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="sequences designed per forward pass (near-free: 8 costs "
                         "0.87 s vs 0.53 s for 1)")
    ap.add_argument("--number-of-batches", type=int, default=1)
    ap.add_argument("--fixed-residues", default="",
                    help='global, e.g. "A45 A46"; per-structure via --fixed-residues-json')
    ap.add_argument("--redesigned-residues", default="")
    ap.add_argument("--fixed-residues-json", help="upstream --fixed_residues_multi JSON")
    ap.add_argument("--redesigned-residues-json", help="upstream --redesigned_residues_multi JSON")
    ap.add_argument("--pack-side-chains", action="store_true",
                    help="also pack and write side chains (slower)")
    ap.add_argument("--extra", action="append", default=[],
                    help="any other run.py flags verbatim, repeatable, e.g. "
                         "--extra '--omit_AA CM' --extra '--ligand_mpnn_use_side_chain_context 1'")
    ap.add_argument("--order", choices=["size", "name", "given"], default="size",
                    help="size = largest first, so a size problem shows up early")
    ap.add_argument("--limit", type=int, help="only the first N items")
    ap.add_argument("--overwrite", action="store_true", help="ignore the manifest and redo everything")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--progress-every", type=int, default=10,
                    help="print a progress line every N items; 1 for small runs")
    args = ap.parse_args()

    if not (args.structures or args.pdb_list or args.pdb_multi):
        ap.error("give structures, --pdb-list, or --pdb-multi")

    out = os.path.abspath(args.out)
    paths, _ = collect_structures(args.structures, args.pdb_list, args.pdb_multi, args.ext)
    if not paths:
        sys.exit("no structures matched")

    if args.checkpoint:
        ckpt = os.path.abspath(args.checkpoint)
    else:
        noise = args.noise or DEFAULT_NOISE[args.model_type]
        ckpt = os.path.join(PARAMS, MODELS[args.model_type][1].format(noise=noise))
    if not os.path.isfile(ckpt):
        avail = sorted(os.path.basename(p) for p in glob.glob(os.path.join(PARAMS, "*.pt")))
        sys.exit(f"checkpoint not found: {ckpt}\navailable: {', '.join(avail)}")
    ckpt_flag = ("--checkpoint_ligand_mpnn" if args.checkpoint
                 else MODELS[args.model_type][0])

    if args.order == "size":
        paths.sort(key=lambda p: -os.path.getsize(p))
    elif args.order == "name":
        paths.sort()

    resolved, dupes = stage_unique_names(paths, os.path.join(out, "stage"))

    jobs = []
    for p in paths:
        name, feed = resolved[p]
        for t in args.temperatures:
            for s in args.seeds:
                uid = f"{name}__T{t:g}__s{s}"
                jobs.append({"uid": uid, "name": name, "source": p, "feed": feed,
                             "temperature": t, "seed": s,
                             "file_ending": f"__T{t:g}__s{s}"})
    if args.limit:
        jobs = jobs[:args.limit]

    n_seq = args.batch_size * args.number_of_batches
    if args.dry_run:
        print(f"{len(paths)} structures x {len(args.temperatures)} temperatures "
              f"x {len(args.seeds)} seeds = {len(jobs)} items, "
              f"{n_seq} sequences each = {len(jobs) * n_seq} sequences")
        print(f"model {args.model_type}  checkpoint {os.path.basename(ckpt)}")
        if dupes:
            print(f"{len(dupes)} basename collision(s) will be staged under unique "
                  f"names, e.g. {sorted(dupes)[0]}")
        print(f"projected: ~{len(jobs) * 1.0 / 60:.1f}-{len(jobs) * 4.0 / 60:.1f} min "
              "on GPU at 1-4 s/item, plus ~10 s startup (small monomers; large or "
              "multi-chain structures are slower -- 2GFB measured at 15 s)")
        for j in jobs[:10]:
            print("  ", j["uid"])
        if len(jobs) > 10:
            print(f"   ... and {len(jobs) - 10} more")
        return 0

    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    manifest_path = os.path.join(out, "manifest.jsonl")
    done = {}
    if os.path.exists(manifest_path) and not args.overwrite:
        with open(manifest_path) as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == "ok":
                    done[r["uid"]] = r
    todo = [j for j in jobs if j["uid"] not in done]
    print(f"{len(jobs)} items, {len(done)} already done, {len(todo)} to run", flush=True)

    sys.path.insert(0, REPO)
    t0 = time.time()
    import torch
    import run as upstream
    parser = upstream_parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} on {device}, startup {time.time() - t0:.1f} s",
          flush=True)
    if device == "cpu":
        print("NOTE: no GPU visible -- pass --nv to apptainer if you meant to use one",
              flush=True)

    prov = {"driver": os.path.basename(__file__),
            "driver_sha256": sha256(os.path.abspath(__file__)),
            "container": os.environ.get("APPTAINER_CONTAINER")
                         or os.environ.get("SINGULARITY_CONTAINER"),
            "repo": REPO, "checkpoint": ckpt,
            "checkpoint_sha256": sha256(ckpt),
            "model_type": args.model_type, "device": device,
            "torch": torch.__version__,
            "n_structures": len(paths), "n_items": len(jobs),
            "sequences_per_item": n_seq,
            "temperatures": args.temperatures, "seeds": args.seeds,
            "extra": args.extra, "argv": sys.argv[1:],
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        prov["repo_commit"] = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
            text=True, timeout=20).stdout.strip() or None
    except Exception:
        prov["repo_commit"] = None
    json.dump(prov, open(os.path.join(out, "run.json"), "w"), indent=1)

    def argv_for(job, batch_size):
        a = ["--model_type", args.model_type, ckpt_flag, ckpt,
             "--pdb_path", job["feed"], "--out_folder", out + "/",
             "--temperature", str(job["temperature"]), "--seed", str(job["seed"]),
             "--batch_size", str(batch_size),
             "--number_of_batches", str(args.number_of_batches),
             "--file_ending", job["file_ending"], "--verbose", "0"]
        if args.fixed_residues:
            a += ["--fixed_residues", args.fixed_residues]
        if args.redesigned_residues:
            a += ["--redesigned_residues", args.redesigned_residues]
        if args.fixed_residues_json:
            a += ["--fixed_residues_multi", os.path.abspath(args.fixed_residues_json)]
        if args.redesigned_residues_json:
            a += ["--redesigned_residues_multi",
                  os.path.abspath(args.redesigned_residues_json)]
        if args.pack_side_chains:
            a += ["--pack_side_chains", "1",
                  "--checkpoint_path_sc", os.path.join(PARAMS, SC_CHECKPOINT)]
        for e in args.extra:
            a += shlex.split(e)
        return a

    n_ok = n_fail = 0
    mf = open(manifest_path, "a")
    for i, job in enumerate(todo, 1):
        rec = {"uid": job["uid"], "name": job["name"], "source": job["source"],
               "temperature": job["temperature"], "seed": job["seed"]}
        buf = io.StringIO()
        it = time.time()
        batch_size = args.batch_size
        for attempt in (1, 2):
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    upstream.main(parser.parse_args(argv_for(job, batch_size)))
                rec.update(status="ok", batch_size=batch_size)
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                oom = "out of memory" in str(e).lower()
                if oom and attempt == 1 and batch_size > 1:
                    # upstream holds no per-item guard; drop the batch and retry
                    # once before giving the item up
                    torch.cuda.empty_cache()
                    batch_size = 1
                    buf.write(f"\n[driver] OOM at batch_size {args.batch_size}, "
                              "retrying at 1\n")
                    continue
                rec.update(status="fail", error=f"{type(e).__name__}: {str(e)[:300]}",
                           oom=bool(oom))
                break
            except Exception as e:
                rec.update(status="fail", error=f"{type(e).__name__}: {str(e)[:300]}")
                buf.write("\n" + traceback.format_exc())
                break
        rec["wall_s"] = round(time.time() - it, 2)
        if device == "cuda":
            # max_memory_allocated is a running maximum since process start, so
            # reset it per item -- otherwise every row inherits the largest
            # structure's peak and the column means nothing
            rec["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        fa = os.path.join(out, "seqs", job["name"] + job["file_ending"] + ".fa")
        rec["fasta"] = fa if os.path.isfile(fa) else None
        if rec["status"] == "ok" and not rec["fasta"]:
            rec.update(status="fail", error="upstream returned without writing a FASTA")
        if rec["status"] == "ok":
            n_ok += 1
        else:
            n_fail += 1
            with open(os.path.join(out, "logs", job["uid"] + ".log"), "w") as lg:
                lg.write(buf.getvalue())
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        if i % args.progress_every == 0 or i == len(todo):
            el = time.time() - t0
            rate = i / el
            print(f"[{i}/{len(todo)}] {job['uid']} {rec['status']} "
                  f"{rec['wall_s']:.1f}s | {rate * 60:.0f} items/min | "
                  f"eta {(len(todo) - i) / rate / 60:.1f} min", flush=True)
    mf.close()

    # ---- aggregate every sequence into one table
    rows = []
    with open(manifest_path) as fh:
        recs = [json.loads(ln) for ln in fh if ln.strip()]
    for r in {x["uid"]: x for x in recs if x.get("status") == "ok"}.values():
        if not r.get("fasta") or not os.path.isfile(r["fasta"]):
            continue
        for name, hdr, seq in parse_fasta(r["fasta"]):
            rows.append({
                "uid": r["uid"], "name": name, "source_pdb": r["source"],
                "is_native": "id" not in hdr,
                "design_index": hdr.get("id", ""),
                "temperature": hdr.get("T", r["temperature"]),
                "seed": hdr.get("seed", r["seed"]),
                "overall_confidence": hdr.get("overall_confidence", ""),
                "ligand_confidence": hdr.get("ligand_confidence", ""),
                "seq_rec": hdr.get("seq_rec", ""),
                "n_chains": seq.count(":") + 1,
                "seq_len": len(seq.replace(":", "")),
                "sequence": seq,
                "fasta": r["fasta"]})
    cols = ["uid", "name", "source_pdb", "is_native", "design_index", "temperature",
            "seed", "overall_confidence", "ligand_confidence", "seq_rec", "n_chains",
            "seq_len", "sequence", "fasta"]
    with open(os.path.join(out, "designs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(out, "designs.fasta"), "w") as fh:
        for r in rows:
            if r["is_native"]:
                continue
            fh.write(f">{r['uid']}|id={r['design_index']}|T={r['temperature']}"
                     f"|seed={r['seed']}|conf={r['overall_confidence']}"
                     f"|lig_conf={r['ligand_confidence']}|seq_rec={r['seq_rec']}\n"
                     f"{r['sequence']}\n")

    n_designs = sum(1 for r in rows if not r["is_native"])
    # totals span every invocation that wrote to this manifest, not just this
    # one -- a resumed run reporting n_ok=0 next to a full designs.csv would
    # read as a failed run
    latest = {x["uid"]: x for x in recs}
    prov.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                wall_s=round(time.time() - t0, 1),
                this_invocation={"n_ok": n_ok, "n_fail": n_fail},
                n_ok=sum(1 for x in latest.values() if x.get("status") == "ok"),
                n_fail=sum(1 for x in latest.values() if x.get("status") != "ok"),
                n_pending=len(jobs) - len(latest),
                n_sequences=n_designs)
    json.dump(prov, open(os.path.join(out, "run.json"), "w"), indent=1)
    print(f"\nthis invocation: {n_ok} ok, {n_fail} failed in "
          f"{(time.time() - t0) / 60:.1f} min\n"
          f"cumulative: {prov['n_ok']}/{len(jobs)} items ok, "
          f"{n_designs} sequences -> {out}/designs.csv", flush=True)
    if n_fail:
        print(f"failures logged under {out}/logs/", flush=True)
    return 1 if n_fail and not n_ok else 0


if __name__ == "__main__":
    sys.exit(main())
