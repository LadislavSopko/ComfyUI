from huggingface_hub import snapshot_download


# Download model to a local directory
model_name = "deepseek-ai/Janus-Pro-7B"
local_dir = "Models\\Janus-Pro\\7B"

snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
print(f"Model {model_name} downloaded successfully!")

