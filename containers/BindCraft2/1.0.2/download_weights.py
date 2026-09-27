#!/usr/bin/env python3
"""
download_weights.py - Download and bake model weights for BindCraft2 (1.0.2).

Downloads:
- AlphaFold 2 parameters: alphafold_params_2022-12-06.tar (Google DeepMind)
  Contains params_model_1_multimer_v3..model_5_multimer_v3, params_model_1_ptm, params_model_2_ptm.

Verifies:
- ProteinMPNN weights shipped in bindcraft/weights/proteinmpnn/:
  weights_neutral, weights_negative, weights_positive.

Generates:
- /app/models/WEIGHTS.json recording sources, file sizes, SHA256 checksums, and licensing.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ALPHAFOLD_PARAMS_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
EXPECTED_AF2_MODELS = [
    "model_1_multimer_v3",
    "model_2_multimer_v3",
    "model_3_multimer_v3",
    "model_4_multimer_v3",
    "model_5_multimer_v3",
    "model_1_ptm",
    "model_2_ptm",
]

MPNN_VARIANTS = ["weights_neutral", "weights_negative", "weights_positive"]
MPNN_CHECKPOINTS = ["v_48_002.npz", "v_48_010.npz", "v_48_020.npz", "v_48_030.npz"]


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_stream(url: str, dest_path: Path, max_retries: int = 5):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + f".tmp.{os.getpid()}")

    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Downloading {url} -> {dest_path} (attempt {attempt}/{max_retries})...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Ribbon-Containers build)"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_path, "wb") as f:
                total_bytes = int(resp.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 4 * 1024 * 1024  # 4MB chunks
                start_time = time.time()
                last_log = start_time
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log >= 10:
                        pct = (downloaded / total_bytes * 100) if total_bytes else 0
                        speed = downloaded / (now - start_time) / (1024 * 1024)
                        logging.info(f"  Downloaded {downloaded / (1024*1024):.1f} MB / {total_bytes / (1024*1024):.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s")
                        last_log = now
            tmp_path.rename(dest_path)
            logging.info(f"Downloaded {dest_path.name} ({dest_path.stat().st_size} bytes)")
            return
        except Exception as e:
            logging.warning(f"Download attempt {attempt} failed: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                logging.error(f"Failed to download {url} after {max_retries} attempts.")
                raise


def download_and_extract_alphafold(models_dir: Path):
    af_dir = models_dir / "alphafold"
    af_dir.mkdir(parents=True, exist_ok=True)

    tar_path = models_dir / "alphafold_params_2022-12-06.tar"
    download_stream(ALPHAFOLD_PARAMS_URL, tar_path)

    logging.info(f"Extracting {tar_path.name} into {af_dir}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(af_dir)
    logging.info("AlphaFold parameters extracted successfully.")

    # Remove tar to save disk space in image
    tar_path.unlink()
    logging.info(f"Removed temporary archive {tar_path.name}.")

    # Validate presence of expected models
    for m in EXPECTED_AF2_MODELS:
        param_file = af_dir / f"params_{m}.npz"
        assert param_file.is_file(), f"Missing expected AlphaFold parameter file: {param_file}"
        assert param_file.stat().st_size >= (100 << 20), f"File {param_file} too small: {param_file.stat().st_size} bytes"
        logging.info(f"Verified AF2 model: {m} ({param_file.stat().st_size / (1024*1024):.1f} MB)")


def setup_and_verify_proteinmpnn(bindcraft_dir: Path, models_dir: Path):
    src_mpnn = bindcraft_dir / "bindcraft" / "weights" / "proteinmpnn"
    dst_mpnn = models_dir / "proteinmpnn"
    dst_mpnn.mkdir(parents=True, exist_ok=True)

    for variant in MPNN_VARIANTS:
        src_var = src_mpnn / variant
        dst_var = dst_mpnn / variant
        dst_var.mkdir(parents=True, exist_ok=True)
        for ckpt in MPNN_CHECKPOINTS:
            src_file = src_var / ckpt
            dst_file = dst_var / ckpt
            assert src_file.is_file(), f"Missing ProteinMPNN source file: {src_file}"
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
            assert dst_file.stat().st_size >= (1 << 20), f"ProteinMPNN file too small: {dst_file}"
            logging.info(f"Verified ProteinMPNN: {variant}/{ckpt} ({dst_file.stat().st_size} bytes)")


def generate_manifest(models_dir: Path, bindcraft_dir: Path):
    manifest = {
        "software": "BindCraft2",
        "version": "1.0.2",
        "commit": "f5275212e48a379626c72a32b7f13129f3ac1417",
        "date_baked": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "models": {
            "alphafold2": {
                "source_url": ALPHAFOLD_PARAMS_URL,
                "license": "Apache 2.0 / CC-BY 4.0",
                "parameters_directory": str(models_dir / "alphafold"),
                "files": {},
            },
            "proteinmpnn": {
                "license": "MIT",
                "weights_directory": str(models_dir / "proteinmpnn"),
                "files": {},
            },
        },
    }

    af_dir = models_dir / "alphafold"
    for p in sorted(af_dir.glob("params_*.npz")):
        manifest["models"]["alphafold2"]["files"][p.name] = {
            "size_bytes": p.stat().st_size,
            "sha256": compute_sha256(p),
        }

    mpnn_dir = models_dir / "proteinmpnn"
    for variant in MPNN_VARIANTS:
        for p in sorted((mpnn_dir / variant).glob("*.npz")):
            key = f"{variant}/{p.name}"
            manifest["models"]["proteinmpnn"]["files"][key] = {
                "size_bytes": p.stat().st_size,
                "sha256": compute_sha256(p),
            }

    weights_json = models_dir / "WEIGHTS.json"
    with open(weights_json, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Generated manifest: {weights_json} ({weights_json.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Bake weights for BindCraft2 container.")
    parser.add_argument("--base_dir", type=str, default="/app", help="Base directory containing models and bindcraft.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    models_dir = base_dir / "models"
    bindcraft_dir = base_dir / "bindcraft"

    logging.info("=== STEP 1: Downloading & Extracting AlphaFold 2 parameters ===")
    download_and_extract_alphafold(models_dir)

    logging.info("=== STEP 2: Verifying & Setting up ProteinMPNN weights ===")
    setup_and_verify_proteinmpnn(bindcraft_dir, models_dir)

    logging.info("=== STEP 3: Generating WEIGHTS.json provenance manifest ===")
    generate_manifest(models_dir, bindcraft_dir)

    logging.info("=== BAKED WEIGHTS COMPLETE ===")


if __name__ == "__main__":
    main()
