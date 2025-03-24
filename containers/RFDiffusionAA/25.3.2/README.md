# LigandMPNN Docker/Apptainer

https://github.com/baker-laboratory/rf_diffusion_all_atom

### Usage
- Docker: Not implemented.
- Apptainer: ``` apptainer run --nv rfdiffusionaa_25.3.2.sif python -u rf_diffusion_all_atom/run_inference.py inference.input_pdb=rf_diffusion_all_atom/input/7v11.pdb contigmap.contigs=[\'150-150\'] inference.ligand=OQO inference.num_designs=1```
- These are the CLI parameters which can be used:
```
== Config ==
Override anything in the config (foo.bar=value)

inference:
  input_pdb: test_data/1qys.pdb
  num_designs: 10
  design_startnum: 0
  ppi_design: false
  autogenerate_contigs: false
  symmetry: null
  recenter: true
  radius: 10.0
  model_only_neighbors: false
  num_recycles: 1
  recycle_schedule: null
  softmax_T: 1.0e-05
  output_prefix: samples/design
  scaffold_guided: false
  model_runner: NRBStyleSelfCond
  cautious: true
  recycle_between: false
  align_motif: false
  autoregressive_confidence: true
  no_confidence: true
  use_jw_selfcond: false
  symmetric_self_cond: true
  final_step: 1
  feed_true_xt: false
  annotate_termini: true
  deterministic: false
  zero_weights: false
  ligand: null
  ema: true
  contig_as_guidepost: false
  remove_guideposts_from_output: true
  state_dict_to_load: model_state_dict
  infer_guidepost_positions: true
  guidepost_xyz_as_design: true
  guidepost_xyz_as_design_bb:
  - false
  ckpt_path: RFDiffusionAA_paper_weights.pt
  str_self_cond: true
contigmap:
  contigs:
  - '20'
  - A3-23
  - '30'
  contig_atoms: null
  inpaint_str: null
  inpaint_seq: null
  length: null
model:
  n_extra_block: 4
  n_main_block: 32
  n_ref_block: 4
  d_msa: 256
  d_msa_full: 64
  d_pair: 192
  d_templ: 64
  n_head_msa: 8
  n_head_pair: 6
  n_head_templ: 4
  d_hidden: 32
  d_hidden_templ: 64
  p_drop: 0.0
  SE3_param:
    num_layers: 1
    num_channels: 32
    num_degrees: 2
    n_heads: 4
    div: 4
    l0_in_features: 64
    l0_out_features: 64
    l1_in_features: 3
    l1_out_features: 2
    num_edge_features: 64
  SE3_ref_param:
    n_extra_block: -1
    n_main_block: -1
    n_ref_block: -1
    n_finetune_block: -1
    d_msa: -1
    d_msa_full: -1
    d_pair: -1
    d_templ: -1
    n_head_msa: -1
    n_head_pair: -1
    n_head_templ: -1
    d_hidden: -1
    d_hidden_templ: -1
    p_drop: -1
    use_extra_l1: -1
    use_atom_frames: -1
    freeze_track_motif: -1
    num_layers: 2
    num_channels: 32
    num_degrees: 2
    l0_in_features: 64
    l0_out_features: 64
    l1_in_features: 3
    l1_out_features: 2
    num_edge_features: 64
    n_heads: 4
    div: 4
  freeze_track_motif: false
  symmetrize_repeats: false
  repeat_length: null
  symmsub_k: null
  sym_method: mean
  copy_main_block_template: false
  main_block: null
  n_finetune_block: 0
  use_chiral_l1: true
  use_lj_l1: true
  use_atom_frames: true
  recycling_type: all
  use_same_chain: true
  enable_same_chain: true
  refiner_topk: 64
  lj_lin: 0.75
diffuser:
  T: 200
  b_0: 0.01
  b_T: 0.07
  schedule_type: linear
  so3_type: igso3
  aa_decode_steps: 40
  chi_type: wrapped_normal
  crd_scale: 0.0667
  schedule_kwargs: {}
  partial_T: null
  so3_schedule_type: linear
  min_b: 1.5
  max_b: 2.5
  min_sigma: 0.02
  max_sigma: 1.5
seq_diffuser:
  s_b0: null
  s_bT: null
  schedule_type: null
  loss_type: null
  seqdiff: null
denoiser:
  noise_scale_ca: 1
  noise_scale_frame: 1
  noise_scale_torsion: 1
ppi:
  hotspot_res: null
  binderlen: null
potentials:
  guiding_potentials: null
  guide_scale: 10
  guide_decay: constant
  olig_inter_all: null
  olig_intra_all: null
  olig_custom_contact: null
contig_settings:
  ref_idx: null
  hal_idx: null
  idx_rf: null
  inpaint_seq_tensor: null
  inpaint_str_tensor: null
preprocess:
  sidechain_input: false
  motif_sidechain_input: true
  sequence_decode: true
  d_t1d: 22
  d_t2d: 44
  prob_self_cond: 0.0
  str_self_cond: false
  seq_self_cond: false
  predict_previous: false
logging:
  inputs: false
```

### Implementation Notes
- This is based on the existing apptainer image (`rf_se3_diffusion.sif`) downloaded with:
    - `wget http://files.ipd.uw.edu/pub/RF-All-Atom/containers/rf_se3_diffusion.sif`
- I pre-download the weights (since apptainer doesn't use nice layering like Docker). Downloaded from:
    - `wget http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFDiffusionAA_paper_weights.pt`

Because of paths in the code, I needed to add a symlink to the paper weights in the current directory. Additionally, I needed to copy the full RFDiffusion code into the current directory (under `./rf_diffusion_all_atom/`). This is because at runtime, the code caches files within the code directory (this is hard-coded), and apptainer is a read-only system. In the future I'd like to clean this up.

### To Do:
- Rebuild to add final line where the copied `./rf_diffusion_all_atom/` directory is deleted.
- Potentially fork code to  clean up paths etc. Low priority.