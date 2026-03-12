import coremltools as ct
import numpy as np
import sys

model_path = "quantized_checkpoints_coreml/acestep-v15-turbo-coreml-16bit.mlpackage"
print("Loading DiT model:", model_path)
model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
for spec_input in model.get_spec().description.input:
    print(spec_input.name)


# inputs for DiT:
# timestep: Float32 [1]
# timestep_r: Float32 [1]
# hidden_states: Float32 [1, 128, 64] (wait, Swift says 750 frames. Let's use 128 as the test)
# encoder_hidden_states: Float32 [1, 90, 2048]
# encoder_attention_mask: Float32 [1, 90]
# context_latents: Float32 [1, 128, 128]
# position_ids: Int32 [1, 128]
# cache_position: Int32 [128]

B = 1
T = 750
C = 64
E = 8

inputs = {
    "timestep": np.array([0.75], dtype=np.float32),
    "timestep_r": np.array([0.75], dtype=np.float32),
    "hidden_states": np.zeros((B, T, C), dtype=np.float32),
    "encoder_hidden_states": np.zeros((B, E, 2048), dtype=np.float32),
    "attention_mask": np.ones((B, T), dtype=np.float32),
    "encoder_attention_mask": np.ones((B, E), dtype=np.float32),
    "context_latents": np.zeros((B, T, 128), dtype=np.float32),
    "position_ids": np.zeros((B, T), dtype=np.int32),
    "cache_position": np.zeros((T,), dtype=np.int32)
}

print("Predicting...")
try:
    out = model.predict(inputs)
    print("Success! Keys:", out.keys())
    print("Output shape:", out[list(out.keys())[0]].shape)
except Exception as e:
    print("Error during predict:", e)
