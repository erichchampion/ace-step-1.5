import coremltools as ct
import numpy as np
import sys
import glob

model_path = "quantized_checkpoints_coreml/Qwen3-Embedding-0.6B-coreml-8bit.mlpackage"
print("Loading model:", model_path)
model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)

# Create inputs just like Swift does
seq_len = 128
input_ids = np.zeros((1, seq_len), dtype=np.int32)

# token IDs for "[Instrumental]" normally is ~151665
sample_tokens = [151665] + [0] * (seq_len - 1)
input_ids[0, :] = sample_tokens

attention_mask = np.ones((1, seq_len), dtype=np.int32)

print("Predicting...")
try:
    outputs = model.predict({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    print("Keys:", outputs.keys())
    hidden_states = outputs["hidden_states"]
    print("Shape:", hidden_states.shape)
    print("First 10 values:", hidden_states[0, 0, :10])
    print("Contains NaN:", np.isnan(hidden_states).any())
    print("Contains Inf:", np.isinf(hidden_states).any())
except Exception as e:
    print("Prediction failed:", e)
