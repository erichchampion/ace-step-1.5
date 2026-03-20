from transformers import AutoModelForCausalLM
import torch
# Just load the config/model structure without weights to test hasattr
model = AutoModelForCausalLM.from_config(AutoModelForCausalLM.config_class())
print("Has model:", hasattr(model, "model"))
