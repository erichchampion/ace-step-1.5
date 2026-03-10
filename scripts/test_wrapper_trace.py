import torch
from transformers import AutoModelForCausalLM

path = "/Users/erich/git/github/erichchampion/audio/ace-step-1.5/checkpoints/Qwen3-Embedding-0.6B"

class CausalMWrapper(torch.nn.Module):
    def __init__(self, m, is_text_encoder):
        super().__init__()
        self.m = m
        self.is_text_encoder = is_text_encoder
    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=False)
        if self.is_text_encoder:
            if hasattr(self.m, "model"):
                return self.m.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            return out[0]
        return out[0]

try:
    model_causal = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32).eval()
    wrapper = CausalMWrapper(model_causal, True).eval()
    
    example_input = (torch.zeros((1, 128), dtype=torch.int32), torch.ones((1, 128), dtype=torch.int32))
    
    # Trace it!
    traced = torch.jit.trace(wrapper, example_input, strict=False)
    out_traced = traced(*example_input)
    print("Traced output shape:", out_traced.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
