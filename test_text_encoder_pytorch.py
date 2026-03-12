import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

import transformers.models.qwen3.modeling_qwen3
def fake_create_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, **kwargs):
    batch_size, seq_len = input_embeds.shape[0], input_embeds.shape[1]
    mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=input_embeds.device)
    mask = torch.tril(mask)
    final_mask = (1.0 - mask.to(input_embeds.dtype)) * torch.finfo(input_embeds.dtype).min
    return final_mask
transformers.models.qwen3.modeling_qwen3.create_causal_mask = fake_create_causal_mask

model_dir = "checkpoints/Qwen3-Embedding-0.6B"
print("Loading model from", model_dir)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()

seq_len = 32
input_ids = torch.zeros((1, seq_len), dtype=torch.int32)
sample_tokens = [151665] + [0] * (seq_len - 1)
input_ids[0, :] = torch.tensor(sample_tokens)

attention_mask = torch.ones((1, seq_len), dtype=torch.int32)

with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    if isinstance(out, tuple):
        hidden_states = out[0]
    else:
        hidden_states = getattr(out, "last_hidden_state", out)
        
    hidden_states = hidden_states.numpy()
    
    print("Shape:", hidden_states.shape)
    print("First 10:", hidden_states[0, 0, :10])
    print("Contains NaN:", np.isnan(hidden_states).any())
