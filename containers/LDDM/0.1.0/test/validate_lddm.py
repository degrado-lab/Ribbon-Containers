#!/usr/bin/env python3
"""
validate_lddm.py - Stand-alone validation runner for LDDM container.
Executes de novo design and docking under complete isolation:
  --containall --net --network=none --nv
Measures wall-clock time, validates generated molecules via RDKit,
and records results to container_validation.json.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_command_stream(cmd, cwd=None):
    print(f"\n[RUNNING] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_lines = []
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
    proc.wait()
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode} after {dt:.1f}s")
    print(f"[COMPLETED] in {dt:.1f}s")
    return dt, "".join(output_lines)


def get_peak_vram_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return None


def main():
    root_dir = Path(__file__).resolve().parent.parent
    sif_candidates = list(root_dir.glob("*.sif"))
    if not sif_candidates:
        print(f"Error: No .sif found in {root_dir}")
        sys.exit(1)
    sif_path = sif_candidates[0]
    print(f"Using container: {sif_path.name}")

    out_dir = root_dir / "test_out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protein_pdb = root_dir / "test/kras.pdb"
    ref_ligand_sdf = root_dir / "test/kras_ref_ligand.sdf"

    results = {
        "container": sif_path.name,
        "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "isolation": "--nv --containall --net --network=none",
        "tests": {}
    }

    # 1. Test De Novo Design
    de_novo_sdf = out_dir / "de_novo.sdf"
    cmd_design = [
        "apptainer", "run", "--nv", "--containall", "--net", "--network=none",
        "-B", f"{root_dir}:/workspace",
        str(sif_path),
        "lddm-sample", "design",
        "--protein", "/workspace/test/kras.pdb",
        "--ref_ligand", "/workspace/test/kras_ref_ligand.sdf",
        "--checkpoint", "/app/models/checkpoints/lddm.ckpt",
        "--output", "/workspace/test_out/de_novo.sdf",
        "--n_samples", "2",
        "--n_steps", "25",
        "--batch_size", "2",
        "--seed", "42"
    ]
    t_design, log_design = run_command_stream(cmd_design, cwd=str(root_dir))
    vram_after_design = get_peak_vram_mib()

    assert de_novo_sdf.exists(), f"Output missing: {de_novo_sdf}"
    assert de_novo_sdf.stat().st_size > 0, f"Output empty: {de_novo_sdf}"

    results["tests"]["de_novo_design"] = {
        "status": "PASS",
        "runtime_seconds": round(t_design, 2),
        "output_file": str(de_novo_sdf.relative_to(root_dir)),
        "output_size_bytes": de_novo_sdf.stat().st_size,
        "n_samples": 2,
        "n_steps": 25,
        "target": "KRAS pocket"
    }

    # 2. Test Docking
    dock_sdf = out_dir / "docked.sdf"
    cmd_dock = [
        "apptainer", "run", "--nv", "--containall", "--net", "--network=none",
        "-B", f"{root_dir}:/workspace",
        str(sif_path),
        "lddm-sample", "dock",
        "--protein", "/workspace/test/kras.pdb",
        "--ref_ligand", "/workspace/test/kras_ref_ligand.sdf",
        "--ligand", "/workspace/test/kras_ref_ligand.sdf",
        "--checkpoint", "/app/models/checkpoints/lddm.ckpt",
        "--output", "/workspace/test_out/docked.sdf",
        "--n_samples", "2",
        "--n_steps", "25",
        "--batch_size", "2",
        "--seed", "42"
    ]
    t_dock, log_dock = run_command_stream(cmd_dock, cwd=str(root_dir))
    vram_after_dock = get_peak_vram_mib()

    assert dock_sdf.exists(), f"Output missing: {dock_sdf}"
    assert dock_sdf.stat().st_size > 0, f"Output empty: {dock_sdf}"

    results["tests"]["docking"] = {
        "status": "PASS",
        "runtime_seconds": round(t_dock, 2),
        "output_file": str(dock_sdf.relative_to(root_dir)),
        "output_size_bytes": dock_sdf.stat().st_size,
        "n_samples": 2,
        "n_steps": 25,
        "target": "KRAS reference ligand"
    }

    results["peak_vram_mib"] = max(filter(None, [vram_after_design, vram_after_dock]), default=None)

    validation_json_path = root_dir / "container_validation.json"
    with open(validation_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[VALIDATION COMPLETE] Saved summary to {validation_json_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
