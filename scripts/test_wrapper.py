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
    model_causal = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32)
    wrapper = CausalMWrapper(model_causal, True)
    out = wrapper(torch.tensor([[0, 1, 2]]), torch.tensor([[1, 1, 1]]))
    print("Wrapper output shape:", out.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
