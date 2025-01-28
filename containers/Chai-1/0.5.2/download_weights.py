from chai_lab.utils.paths import chai1_component, downloads_path
import chai_lab.utils.paths as paths
from transformers import EsmTokenizer, EsmModel  
 
components = ["bond_loss_input_proj", "feature_embedding", "token_embedder", "trunk", "diffusion_module", "confidence_head"]

#download model weights
for component in components:
    chai1_component(component + ".pt")
 
#download cached rdkit conformers
paths.cached_conformers.get_path()
 
#download ESM tokenizer and model
esm_cache_folder = downloads_path.joinpath("esm")
model_name = "facebook/esm2_t36_3B_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name, cache_dir=esm_cache_folder)
model = EsmModel.from_pretrained(model_name, cache_dir=esm_cache_folder)