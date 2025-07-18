from pathlib import Path

import numpy as np
import torch

from chai_lab.chai1 import run_inference

# We use fasta-like format for inputs.
# Every record may encode protein, ligand, RNA or DNA
#  see example below

example_fasta = """
>protein|example-protein
AGSHSMRYF
""".strip()

fasta_path = Path("/tmp/example.fasta")
fasta_path.write_text(example_fasta)

output_dir = Path("/tmp/outputs")
output_cif_paths = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    # 'default' setup
    num_trunk_recycles=1,       # Minimal run just to download weights.
    num_diffn_timesteps=1,      # 
    seed=42,
    device=torch.device("cpu"),
    use_esm_embeddings=True,
)

# Load pTM, ipTM, pLDDTs and clash scores for sample 2
scores = np.load(output_dir.joinpath("scores.model_idx_2.npz"))
