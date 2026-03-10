from transformers import AutoModelForCausalLM, AutoConfig
import torch

path = "/Users/erich/git/github/erichchampion/audio/ace-step-1.5/checkpoints/Qwen3-Embedding-0.6B"
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, config=config, trust_remote_code=True, torch_dtype=torch.float32)

print("Modules:")
for name, module in model.named_children():
    print(f"- {name}: {type(module).__name__}")
