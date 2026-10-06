"""Bake the ESMFold2 weights into the image, stored bf16.

Run inside the builder stage:  python download_weights.py --out /app/models

Three repos are needed and all three are ungated (no HF token, unlike ESM3):

  biohub/ESMC-6B        the 6B protein language model, 25.41 GB as published
  biohub/ESMFold2       the full folding trunk, 1.36 GB
  biohub/ESMFold2-Fast  the distilled trunk, 0.76 GB

All three are published with fp32 tensors and all three are cast here.

ESMC-6B is not optional: both trunk configs carry `esmc_id: "biohub/ESMC-6B"`,
and ESMFold2Model.from_pretrained() calls load_esmc(config.esmc_id) on the way
out. One copy serves both trunks.

Two things happen here beyond downloading:

1. Every floating-point tensor is cast to bfloat16. load_esmc()'s signature is
   `precision: str = "bf16"` and the trunks are loaded with
   torch_dtype=torch.bfloat16, so all of these checkpoints are cast to bf16 at
   load time in any normal run — storing them bf16 is bit-identical at inference
   and halves the image. Shards are fetched, cast and deleted one at a time, so
   peak build disk is ~19 GB rather than the ~37 GB a download-then-convert
   would need.

2. `esmc_id` in both trunk configs is rewritten from the repo id to the in-image
   path. Without that, from_pretrained() resolves ESMC through the HF cache, and
   a cache lookup on Apptainer's read-only filesystem is exactly what stopped
   weights being baked into the ESM3 container.
"""
import argparse
import json
import os
import shutil

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

REPOS = ["biohub/ESMC-6B", "biohub/ESMFold2", "biohub/ESMFold2-Fast"]
SKIP = {".gitattributes"}
FLOATS = (torch.float64, torch.float32, torch.float16)


def convert_shard(src, dst):
    """Load one safetensors shard, cast float tensors to bf16, write it back."""
    tensors, meta, n_cast = {}, None, 0
    with safe_open(src, framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            t = f.get_tensor(k)
            if t.dtype in FLOATS:
                t = t.to(torch.bfloat16)
                n_cast += 1
            tensors[k] = t.contiguous()
    save_file(tensors, dst, metadata=meta or {"format": "pt"})
    return n_cast, len(tensors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/app/models")
    args = ap.parse_args()

    api = HfApi()
    tmp = "/tmp/hf_dl"
    os.makedirs(tmp, exist_ok=True)
    record = {"stored_dtype": "bfloat16", "models": {}}

    for repo in REPOS:
        name = repo.split("/")[-1]
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        info = api.model_info(repo)
        files = [f for f in api.list_repo_files(repo) if f not in SKIP]
        print(f"=== {repo} @ {info.sha} -> {out} ({len(files)} files)", flush=True)

        bytes_in = bytes_out = 0
        for fn in sorted(files):
            if fn.endswith(".safetensors"):
                src = hf_hub_download(repo, fn, local_dir=tmp)
                dst = os.path.join(out, fn)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                n_cast, n_tot = convert_shard(src, dst)
                bytes_in += os.path.getsize(src)
                bytes_out += os.path.getsize(dst)
                os.remove(src)          # keep peak build disk down
                print(f"    {fn}: {n_cast}/{n_tot} tensors -> bf16", flush=True)
            else:
                p = hf_hub_download(repo, fn, local_dir=out)
                bytes_in += os.path.getsize(p)
                bytes_out += os.path.getsize(p)
                print(f"    {fn}: copied", flush=True)

        # index total_size describes the tensor payload, which just halved
        idx = os.path.join(out, "model.safetensors.index.json")
        if os.path.exists(idx):
            j = json.load(open(idx))
            shards = {v for v in j["weight_map"].values()}
            j.setdefault("metadata", {})["total_size"] = sum(
                os.path.getsize(os.path.join(out, s)) for s in shards)
            json.dump(j, open(idx, "w"), indent=1)

        # Point esmc_id at the local copy. The `dtype` field is deliberately
        # left alone: the published trunk checkpoints are already bf16 while
        # their configs read "float32", and that combination is what the
        # validated runs used, with the runner passing torch_dtype explicitly.
        cfg_path = os.path.join(out, "config.json")
        cfg = json.load(open(cfg_path))
        if "esmc_id" in cfg:
            cfg["esmc_id"] = os.path.join(args.out, "ESMC-6B")
            json.dump(cfg, open(cfg_path, "w"), indent=1)

        # hf_hub_download leaves a metadata dir under local_dir; not wanted in the image
        shutil.rmtree(os.path.join(out, ".cache"), ignore_errors=True)

        record["models"][name] = dict(
            repo=repo, revision=info.sha, gated=False,
            source_bytes=bytes_in, stored_bytes=bytes_out,
            recast_to_bf16=bytes_out < bytes_in * 0.9,
            config_dtype_field=cfg.get("dtype", cfg.get("torch_dtype")),
            esmc_id=cfg.get("esmc_id"),
            files=sorted(os.listdir(out)))
        print(f"    {name}: {bytes_in/1e9:.2f} GB -> {bytes_out/1e9:.2f} GB", flush=True)

    shutil.rmtree(tmp, ignore_errors=True)
    record["total_stored_bytes"] = sum(
        m["stored_bytes"] for m in record["models"].values())
    with open(os.path.join(args.out, "WEIGHTS.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print("BAKED " + json.dumps({k: round(v["stored_bytes"] / 1e9, 2)
                                 for k, v in record["models"].items()}), flush=True)


if __name__ == "__main__":
    main()
