from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
import torch
import os

# Download model to a local directory
model_name = "deepseek-ai/Janus-Pro-7B"
local_dir = "C:\\DeepLearning\\Models\\hugghing_face"

snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
print(f"Model {model_name} downloaded successfully!")

# Load model
print(f"Loading model from {model_name} ...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Convert and save as .safetensors
output_path = os.path.join(model_name, "model.safetensors")
save_file(model.state_dict(), output_path)

print(f"✅ Model converted and saved at: {output_path}")
