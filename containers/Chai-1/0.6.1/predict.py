from pathlib import Path
import numpy as np
import torch
from chai_lab.chai1 import run_inference
 
fasta_path = Path("./my_system.fasta")
 
output_dir = Path("./outputs1")
output_cif_paths = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    # 'default' setup
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    device=torch.device("cuda:0"),
    use_esm_embeddings=True,
)
