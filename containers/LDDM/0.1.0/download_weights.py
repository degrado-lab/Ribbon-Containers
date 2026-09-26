#!/usr/bin/env python3
"""
download_weights.py - Download and bake model checkpoints and reference assets for LDDM.

Downloads:
- Main model checkpoint: lddm.ckpt (CC-BY-NC 4.0)
- MIT model checkpoint: lddm_CDBB.ckpt (MIT)
- 3D validity reference geometry: ligands.sdf
- SynSpace chemical space data (building blocks, reactions, mappings)

Also precompiles reference geometry kernel densities so the Apptainer read-only SIF
does not attempt runtime disk writes to /app/data/validity3d.
Emits /app/models/WEIGHTS.json for auditability and container %test verification.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ZENODO_BASE = "https://zenodo.org/records/22754501/files"

DOWNLOADS = [
    {
        "key": "lddm.ckpt",
        "url": f"{ZENODO_BASE}/lddm.ckpt",
        "subpath": "models/checkpoints/lddm.ckpt",
        "category": "checkpoint",
        "license": "CC-BY-NC 4.0",
        "description": "CD+BB+BN - Main model used for experiments in paper",
    },
    {
        "key": "lddm_CDBB.ckpt",
        "url": f"{ZENODO_BASE}/lddm_CDBB.ckpt",
        "subpath": "models/checkpoints/lddm_CDBB.ckpt",
        "category": "checkpoint",
        "license": "MIT",
        "description": "CD+BB - Model trained without BindingNet",
    },
    {
        "key": "ligands.sdf",
        "url": f"{ZENODO_BASE}/ligands.sdf",
        "subpath": "data/validity3d/ligands.sdf",
        "category": "reference_geometry",
        "license": "Open",
        "description": "3D validity geometry reference ligands for distribution compilation",
    },
    {
        "key": "building_blocks.csv",
        "url": f"{ZENODO_BASE}/building_blocks.csv",
        "subpath": "data/synspace/building_blocks.csv",
        "category": "synspace",
        "license": "Open",
        "description": "SynSpace chemical space building blocks CSV",
    },
    {
        "key": "building_blocks.pkl",
        "url": f"{ZENODO_BASE}/building_blocks.pkl",
        "subpath": "data/synspace/building_blocks.pkl",
        "category": "synspace",
        "license": "Open",
        "description": "SynSpace chemical space building blocks PKL",
    },
    {
        "key": "reactions.json",
        "url": f"{ZENODO_BASE}/reactions.json",
        "subpath": "data/synspace/reactions.json",
        "category": "synspace",
        "license": "Open",
        "description": "SynSpace chemical space reactions JSON",
    },
    {
        "key": "reaction_to_building_blocks.csv",
        "url": f"{ZENODO_BASE}/reaction_to_building_blocks.csv",
        "subpath": "data/synspace/reaction_to_building_blocks.csv",
        "category": "synspace",
        "license": "Open",
        "description": "SynSpace reaction to building blocks CSV",
    },
    {
        "key": "reaction_to_building_blocks.pkl",
        "url": f"{ZENODO_BASE}/reaction_to_building_blocks.pkl",
        "subpath": "data/synspace/reaction_to_building_blocks.pkl",
        "category": "synspace",
        "license": "Open",
        "description": "SynSpace reaction to building blocks PKL",
    },
]


def download_file(url: str, dest_path: Path, max_retries: int = 5):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Downloading {url} -> {dest_path} (attempt {attempt}/{max_retries})...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Ribbon-Containers build)"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as f:
                total_bytes = int(resp.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                start_time = time.time()
                last_log = start_time
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log >= 10:  # log every 10 seconds
                        pct = (downloaded / total_bytes * 100) if total_bytes else 0
                        speed = downloaded / (now - start_time) / (1024 * 1024)
                        logging.info(f"  {downloaded / (1024*1024):.1f} MB / {total_bytes / (1024*1024):.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s")
                        last_log = now
            tmp_path.rename(dest_path)
            logging.info(f"Successfully downloaded {dest_path.name} ({dest_path.stat().st_size} bytes)")
            return
        except Exception as e:
            logging.warning(f"Download failed: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < max_retries:
                sleep_sec = 2 ** attempt
                logging.info(f"Retrying in {sleep_sec} seconds...")
                time.sleep(sleep_sec)
            else:
                logging.error(f"Failed to download {url} after {max_retries} attempts.")
                raise


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def precompile_geometry_reference(ligands_sdf: Path, limit: int = 5000):
    """
    Precompile ligands_geometry_values.p and ligands_geometry_kernel_densities.p
    in the same directory as ligands.sdf. This prevents runtime Errno 30 on read-only SIF.
    Default limit=5000 samples over 130,000 bonds and 200,000 angles, saturating empirical
    distributions for all standard organic chemistry patterns in ~10 minutes.
    """
    logging.info(f"Precompiling reference distributions for {ligands_sdf} (limit={limit})...")
    try:
        from lddm.sbdd_metrics.validity_3d.reference_geometry import ReferenceGeometry
        from lddm.sbdd_metrics.validity_3d.sdf_source import SDFSource
        
        name = ligands_sdf.name.replace(".sdf", "")
        root = str(ligands_sdf.parent)
        ref_geom = ReferenceGeometry(
            source=SDFSource(ligands_path=str(ligands_sdf), name=name, limit=limit),
            root=root,
            minimum_pattern_values=50,
        )
        values_p = ligands_sdf.parent / f"{name}_geometry_values.p"
        kde_p = ligands_sdf.parent / f"{name}_geometry_kernel_densities.p"
        if values_p.exists() and kde_p.exists():
            logging.info(f"Precompiled geometry cache created successfully: {values_p} ({values_p.stat().st_size} B), {kde_p} ({kde_p.stat().st_size} B)")
        else:
            logging.warning(f"Precompilation completed but expected files were not found: {values_p}, {kde_p}")
    except Exception as e:
        logging.error(f"Error during geometry reference precompilation: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Download model weights and datasets for LDDM.")
    parser.add_argument("--base_dir", type=str, default="/app", help="Base directory containing models and data.")
    parser.add_argument("--geometry_limit", type=int, default=5000, help="Max ligands to parse for reference geometry KDE (default 5000).")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    manifest = {
        "zenodo_record": "22754501",
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "geometry_limit": args.geometry_limit,
        "files": {},
    }

    for item in DOWNLOADS:
        target_path = base_dir / item["subpath"]
        download_file(item["url"], target_path)
        sha = compute_sha256(target_path)
        size = target_path.stat().st_size
        manifest["files"][item["key"]] = {
            "path": str(target_path),
            "url": item["url"],
            "sha256": sha,
            "size_bytes": size,
            "category": item["category"],
            "license": item.get("license", "Unknown"),
            "description": item["description"],
        }
        logging.info(f"Recorded {item['key']}: {size} bytes, sha256={sha[:12]}...")

    # Precompile 3D geometry reference
    ligands_sdf = base_dir / "data/validity3d/ligands.sdf"
    if ligands_sdf.exists():
        precompile_geometry_reference(ligands_sdf, limit=args.geometry_limit)
        for cache_name in ["ligands_geometry_values.p", "ligands_geometry_kernel_densities.p"]:
            cache_p = ligands_sdf.parent / cache_name
            if cache_p.exists():
                manifest["files"][cache_name] = {
                    "path": str(cache_p),
                    "size_bytes": cache_p.stat().st_size,
                    "sha256": compute_sha256(cache_p),
                    "category": "precompiled_cache",
                    "description": f"Precompiled KDE cache for {cache_name}",
                }

    weights_json_path = base_dir / "models/WEIGHTS.json"
    weights_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(weights_json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Wrote provenance manifest to {weights_json_path}")
    print("=== BAKED WEIGHTS COMPLETE ===")


if __name__ == "__main__":
    main()
