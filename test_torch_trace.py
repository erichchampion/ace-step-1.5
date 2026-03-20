import torch
import numpy as np

# Load the dummy trace directly
from transformers import AutoModel
import os
import sys

# Append the huggingface cache path so we can import the model
sys.path.insert(0, '/Users/erich/.cache/huggingface/modules/transformers_modules/acestep_hyphen_v15_hyphen_turbo')
import modeling_acestep_v15_turbo
from modeling_acestep_v15_turbo import AceStepConditionGenerationModel

print("Loading model...")
model = AceStepConditionGenerationModel.from_pretrained('/Users/erich/git/github/erichchampion/audio/ace-step-1.5/checkpoints/acestep-v15-turbo', torch_dtype=torch.float32)
model.eval()

# We need to test the decoder directly
decoder = model.decoder
decoder.eval()

# Trace it like quantize_checkpoints.py does
dtype = torch.float32
device = 'cpu'
kwargs_trace = {
    'hidden_states': torch.zeros((1, 64, 64), dtype=dtype).to(device),
    'timestep': torch.tensor([0.75], dtype=dtype).to(device),
    'timestep_r': torch.tensor([0.75], dtype=dtype).to(device),
    'attention_mask': torch.ones((1, 64), dtype=dtype).to(device),
    'encoder_hidden_states': torch.zeros((1, 100, 2048), dtype=dtype).to(device),
    'encoder_attention_mask': torch.ones((1, 100), dtype=dtype).to(device),
    'context_latents': torch.zeros((1, 64, 128), dtype=dtype).to(device),
    'position_ids': torch.arange(64, dtype=torch.int32).unsqueeze(0).to(device),
    'cache_position': torch.arange(64, dtype=torch.int32).to(device),
    'use_cache': False
}

print("Tracing...")
from torch.jit import trace
# Patch it first like quantize_checkpoints
import modeling_acestep_v15_turbo
def patched_create_4d_mask(seq_len, dtype, device, attention_mask=None, sliding_window=None, is_sliding_window=False, is_causal=True, mask_len=None):
    if not is_causal:
        if attention_mask is not None:
            p_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            zero = torch.tensor(0.0, dtype=dtype, device=device)
            inf_val = torch.tensor(-65500.0, dtype=dtype, device=device)
            return torch.where(p_mask, zero, inf_val)
        else:
            return torch.zeros((1, 1, 1, 1), dtype=dtype, device=device)
    return modeling_acestep_v15_turbo.create_4d_mask(int(seq_len), dtype, device, attention_mask, sliding_window, is_sliding_window, is_causal, None if mask_len is None else int(mask_len))
modeling_acestep_v15_turbo.create_4d_mask = patched_create_4d_mask

class AcestepWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, hidden_states, timestep, timestep_r, attention_mask, encoder_hidden_states, encoder_attention_mask, context_latents, position_ids, cache_position):
        kwargs = {
            'hidden_states': hidden_states, 'timestep': timestep, 'timestep_r': timestep_r,
            'attention_mask': attention_mask, 'encoder_hidden_states': encoder_hidden_states,
            'encoder_attention_mask': encoder_attention_mask, 'context_latents': context_latents,
            'position_ids': position_ids, 'cache_position': cache_position, 'use_cache': False
        }
        kwargs['cache_state'] = None
        kwargs['return_dict'] = False
        return self.m.decoder(**kwargs)

wrapper = AcestepWrapper(model).eval()

traced_model = trace(wrapper, (
    kwargs_trace['hidden_states'], kwargs_trace['timestep'], kwargs_trace['timestep_r'],
    kwargs_trace['attention_mask'], kwargs_trace['encoder_hidden_states'],
    kwargs_trace['encoder_attention_mask'], kwargs_trace['context_latents'],
    kwargs_trace['position_ids'], kwargs_trace['cache_position']
))

print("Testing dynamic execution...")
kwargs_run = {
    'hidden_states': torch.zeros((1, 750, 64), dtype=dtype).to(device),
    'timestep': torch.tensor([0.75], dtype=dtype).to(device),
    'timestep_r': torch.tensor([0.75], dtype=dtype).to(device),
    'attention_mask': torch.ones((1, 750), dtype=dtype).to(device),
    'encoder_hidden_states': torch.zeros((1, 8, 2048), dtype=dtype).to(device),
    'encoder_attention_mask': torch.ones((1, 8), dtype=dtype).to(device),
    'context_latents': torch.zeros((1, 750, 128), dtype=dtype).to(device),
    'position_ids': torch.arange(750, dtype=torch.int32).unsqueeze(0).to(device),
    'cache_position': torch.arange(750, dtype=torch.int32).to(device)
}

try:
    out = traced_model(
        kwargs_run['hidden_states'], kwargs_run['timestep'], kwargs_run['timestep_r'],
        kwargs_run['attention_mask'], kwargs_run['encoder_hidden_states'],
        kwargs_run['encoder_attention_mask'], kwargs_run['context_latents'],
        kwargs_run['position_ids'], kwargs_run['cache_position']
    )
    print("Success! output shape:", out[0].shape)
except Exception as e:
    import traceback
    traceback.print_exc()
