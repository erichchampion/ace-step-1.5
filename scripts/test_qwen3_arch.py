import sys
import traceback
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel

path = "/Users/erich/git/github/erichchampion/audio/ace-step-1.5/checkpoints/Qwen3-Embedding-0.6B"

try:
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    print("Config type:", type(config).__name__)

    try:
        model_causal = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32)
        print("AutoModelForCausalLM loaded. Type:", type(model_causal).__name__)
        print("Has .model?", hasattr(model_causal, "model"))
        print("Has .transformer?", hasattr(model_causal, "transformer"))
    except Exception as e:
        print("AutoModelForCausalLM failed:")
        traceback.print_exc()

    try:
        model_auto = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32)
        print("AutoModel loaded. Type:", type(model_auto).__name__)
        print("Has .model?", hasattr(model_auto, "model"))
        print("Has .transformer?", hasattr(model_auto, "transformer"))
        
        # Test forward
        out = model_auto(input_ids=torch.tensor([[0, 1, 2]]), attention_mask=torch.tensor([[1, 1, 1]]))
        if isinstance(out, tuple):
            print("AutoModel out len:", len(out))
            print("AutoModel out[0] shape:", out[0].shape)
        else:
            print("AutoModel out dict keys:", out.keys())
            if hasattr(out, "last_hidden_state"):
                print("AutoModel out.last_hidden_state shape:", out.last_hidden_state.shape)
    except Exception as e:
        print("AutoModel failed:")
        traceback.print_exc()
except Exception as e:
    print("Config failed:")
    traceback.print_exc()
