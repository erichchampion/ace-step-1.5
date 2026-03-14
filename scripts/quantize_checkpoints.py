#!/usr/bin/env python3
"""
Script to convert models in the checkpoints/ directory to Core ML format 
and apply 8-bit, 6-bit, and 4-bit weight palettization (quantization).
Parallel directories will be created for each quantized model.
"""

import os
import sys
import gc
import sympy.printing
import argparse
import traceback
import math
import shutil
import glob
import time
import datetime
from pathlib import Path
from typing import Optional, List, Union, Tuple

def main():
    import os
    import glob
    parser = argparse.ArgumentParser(description="Quantize models to Core ML format.")
    parser.add_argument("model_name", type=str, nargs="?", help="Optional: Name of a specific model to quantize.")
    args = parser.parse_args()

    def log(msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)

    # Flush Hugging Face's transformers modules cache for acestep to ensure local patches apply
    try:
        cache_pattern = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/acestep*")
        for cache_dir in glob.glob(cache_pattern):
            shutil.rmtree(cache_dir, ignore_errors=True)
            log(f"Flushed local huggingface cache: {cache_dir}")
    except Exception as e:
        log(f"Warning: Failed to flush huggingface cache: {e}")

    try:
        import torch
        import coremltools as ct
        import numpy as np
        from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer
        import warnings
        
        def trace_with_stack(model, example_input, **kwargs):
            try:
                if isinstance(example_input, tuple):
                    traced = torch.jit.trace(model, example_input, **kwargs)
                else:
                    traced = torch.jit.trace(model, (example_input,), **kwargs)
                    
                log(f"====== PYTORCH JIT TRACE GRAPH ======\n{str(traced.graph)[:50000]}\n")
                log(f"====== PYTORCH JIT TRACE CODE ======\n{str(traced.code)}\n")
                
                return traced
            except Exception as e:
                log(f"\n[FATAL TRACER ERROR] {type(e).__name__}: {str(e)}")
                traceback.print_exc(file=sys.stdout)
                raise e

    except ImportError as e:
        log(f"Error: Missing required packages. {e}")
        log("Please ensure your dependencies are installed.")
        log("Suggestion: run 'uv pip install -r requirements.txt'")
        sys.exit(1)

    # Apply PyTorch Tracer Bypass Patch for F.embedding
    import inspect
    import textwrap
    import torch.nn.functional as F
    
    if not getattr(F, "_embedding_patched", False):
        try:
            emb_src = inspect.getsource(F.embedding)
            emb_src = emb_src.replace("assert padding_idx < weight.size(0)", "assert True")
            emb_src = emb_src.replace("assert padding_idx >= -weight.size(0)", "assert True")
            local_env = {}
            exec(textwrap.dedent(emb_src), F.__dict__, local_env)
            F.embedding = local_env["embedding"]
            F._embedding_patched = True
        except Exception as e:
            log(f"Failed to patch F.embedding: {e}")

    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        log(f"Error: Directory '{checkpoints_dir}' does not exist.")
        sys.exit(1)

    quantized_dir = Path("quantized_checkpoints_coreml")
    quantized_dir.mkdir(exist_ok=True)

    log(f"Scanning checkpoints directory for models at {checkpoints_dir}...\n")
    
    # Force transformers to never use vmap for sdpa masks during tracing
    try:
        import transformers.masking_utils
        def fake_create_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, **kwargs):
            batch_size, seq_len = input_embeds.shape[0], input_embeds.shape[1]
            mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=input_embeds.device)
            mask = torch.tril(mask)
            final_mask = (1.0 - mask.to(input_embeds.dtype)) * -65500.0
            return final_mask
        transformers.masking_utils.create_causal_mask = fake_create_causal_mask
        if hasattr(transformers.models, "qwen3"):
            transformers.models.qwen3.modeling_qwen3.create_causal_mask = fake_create_causal_mask
    except Exception as e:
        log(f"Warning: could not patch create_causal_mask: {e}")

    # Globally patch repeat_kv to avoid dynamic shape slicing that breaks CoreML tracing
    try:
        def patched_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
            if n_rep == 1:
                return hidden_states
            # Expand only the new n_rep dimension (index 2) and flatten to evade aten::Int
            hidden_states = hidden_states[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1)
            return hidden_states.flatten(1, 2)
            
        import transformers.integrations.sdpa_attention as sdpa_integration
        sdpa_integration.repeat_kv = patched_repeat_kv
        
        import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
        qwen3_modeling.repeat_kv = patched_repeat_kv
        
        def patched_rotate_half(x):
            half = torch.chunk(x, 2, dim=-1)
            return torch.cat((-half[1], half[0]), dim=-1)
        qwen3_modeling.rotate_half = patched_rotate_half
        
        log("  [Global Patch] Applied repeat_kv flatten(1, 2) patch for fully dynamic sequences!")
    except Exception as e:
        log(f"  [Global Patch Warning] Failed to patch repeat_kv: {e}")

    # Iterate over all directories in checkpoints/
    for item in sorted(checkpoints_dir.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue

        if args.model_name and item.name != args.model_name:
            continue
            
        if "bit" in item.name or "coreml" in item.name:
            continue

        log(f"\n--- Processing Model: {item.name} ---")

        bits_to_process = []
        for bits in [16, 8, 6, 4]:
            output_path = quantized_dir / f"{item.name}-coreml-{bits}bit.mlpackage"
            if not output_path.exists():
                bits_to_process.append(bits)
            else:
                log(f"  [{bits}-bit] Output '{output_path}' already exists. Skipping.")
                
        if not bits_to_process:
            log("  All requested quantizations already exist. Skipping to next model.\n")
            continue

        try:
            if "acestep" in item.name.lower():
                dependency_script = Path(__file__).parent / "quantizer_torchao.py"
                target_script = item / "quantizer_torchao.py"
                if dependency_script.exists():
                    shutil.copy(dependency_script, target_script)
                
                modeling_script = Path(__file__).parent.parent / "acestep/models/turbo/modeling_acestep_v15_turbo.py"
                target_modeling = item / "modeling_acestep_v15_turbo.py"
                if modeling_script.exists():
                    shutil.copy(modeling_script, target_modeling)
                    log(f"  [Sync] Copied latest modeling script to {item.name}")
            
            gc.collect()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                    
            log("  Loading PyTorch model into memory (this may take a while)...")
            
            if item.name == "vae":
                from diffusers import AutoencoderOobleck
                from torch.nn.utils.parametrize import remove_parametrizations, is_parametrized
                model = AutoencoderOobleck.from_pretrained(str(item), torch_dtype=torch.float32)
                model.eval()
                
                class VAEWrapper(torch.nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m
                    def forward(self, latents):
                        return self.m.decoder(latents)
                wrapped_model = VAEWrapper(model).eval()
                
                def remove_all_weight_norms(m):
                    for name, child in m.named_children():
                        if is_parametrized(child, "weight"):
                            remove_parametrizations(child, "weight")
                        remove_all_weight_norms(child)
                remove_all_weight_norms(wrapped_model)
                
                # Align shapes to 128 to match the 5.0 second Swift test length
                # Trace with a small example input for fast PyTorch graph construction
                if "STATIC_SEQ_LEN" in os.environ:
                    del os.environ["STATIC_SEQ_LEN"]
                b, c, t = 1, 64, 128
                example_input = (torch.randn((b, c, t), dtype=torch.float32),)
                inputs_schema = [ct.TensorType(name="latents", shape=(1, 64, ct.RangeDim(16, 32768, default=128)), dtype=np.float32)]
                outputs_schema = [ct.TensorType(name="audio")]
                
            else:
                try:
                    config = AutoConfig.from_pretrained(str(item), trust_remote_code=True)
                    if type(config).__name__ == "AceStepConfig":
                        model = AutoModel.from_pretrained(str(item), trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager")
                    else:
                        model = AutoModelForCausalLM.from_pretrained(str(item), trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager")
                except Exception as inner_e:
                    log(f"  [Warning] AutoModel loading failed ({inner_e}), trying generic AutoModel fallback...")
                    model = AutoModel.from_pretrained(str(item), trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager")
                
                model.eval()
                if hasattr(model.config, "use_cache"):
                    model.config.use_cache = False
                
                is_acestep = "AceStep" in type(model).__name__
                if is_acestep:
                    if hasattr(model, 'encoder') and hasattr(model.encoder, 'timbre_encoder'):
                        timbre_cls = type(model.encoder.timbre_encoder)
                        def patched_unpack(self, timbre_embs_packed, refer_audio_order_mask):
                            N, d = timbre_embs_packed.shape
                            timbre_embs_unpack = timbre_embs_packed.unsqueeze(0)
                            new_mask = torch.ones((1, N), dtype=torch.long, device=timbre_embs_packed.device)
                            return timbre_embs_unpack, new_mask
                        timbre_cls.unpack_timbre_embeddings = patched_unpack
                        log("  [Patch] Applied unpack_timbre_embeddings patch.")

                    # Patch create_4d_mask to avoid torch.arange int() casts during JIT trace!
                    acestep_module_name = type(model).__module__
                    import sys
                    acestep_module = sys.modules[acestep_module_name]
                    original_create_4d_mask = acestep_module.create_4d_mask

                    # Mutable closure dict to pass sliding window mask from wrapper to create_4d_mask
                    _sliding_mask_state = {}

                    def patched_create_4d_mask(seq_len, dtype, device, attention_mask=None, sliding_window=None, is_sliding_window=False, is_causal=True, mask_len=None):
                        if mask_len is None:
                            mask_len = seq_len
                            
                        # If simple unmasked bidirectional attention WITHOUT sliding window,
                        # just return zeros (all positions attend to all positions).
                        # CRITICAL: Do NOT shortcut when is_sliding_window=True — the sliding
                        # window mask restricts attention to |i-j| <= window positions, which
                        # is essential for correct output at long sequences (e.g., 30s audio).
                        if not is_causal and not is_sliding_window:
                            if attention_mask is not None:
                                p_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
                                # p_mask is [B, 1, 1, mask_len]
                                zero = torch.tensor(0.0, dtype=dtype, device=device)
                                inf_val = torch.tensor(-65500.0, dtype=dtype, device=device)
                                return torch.where(p_mask, zero, inf_val)
                            else:
                                return torch.zeros((1, 1, 1, 1), dtype=dtype, device=device)
                        
                        # Sliding window: use the pre-computed mask passed from the wrapper.
                        # This avoids ALL dynamic ops (arange, triu, cumsum) in the trace graph.
                        if is_sliding_window and not is_causal and 'mask' in _sliding_mask_state:
                            return _sliding_mask_state['mask']
                        
                        # For causal or other cases, fall back to the original function
                        return original_create_4d_mask(int(seq_len), dtype, device, attention_mask, sliding_window, is_sliding_window, is_causal, None if mask_len is None else int(mask_len))
                    
                    acestep_module.create_4d_mask = patched_create_4d_mask
                    log("  [Patch] Applied create_4d_mask patch to bypass JIT int() cast bugs.")

                    class AcestepWrapper(torch.nn.Module):
                        def __init__(self, m):
                            super().__init__()
                            self.m = m
                            self.is_turbo = getattr(m.config, 'is_turbo', False)
                        def forward(self, hidden_states, timestep, timestep_r, attention_mask, encoder_hidden_states, encoder_attention_mask, context_latents, position_ids, cache_position, sliding_window_mask):
                            # Store the pre-computed sliding window mask in the closure
                            # so patched_create_4d_mask can read it for sliding attention layers
                            _sliding_mask_state['mask'] = sliding_window_mask
                            kwargs = {
                                'hidden_states': hidden_states, 'timestep': timestep, 'timestep_r': timestep_r,
                                'attention_mask': attention_mask, 'encoder_hidden_states': encoder_hidden_states,
                                'encoder_attention_mask': encoder_attention_mask, 'context_latents': context_latents,
                                'position_ids': position_ids, 'cache_position': cache_position,
                                'use_cache': False
                            }
                            if self.is_turbo:
                                kwargs['cache_state'] = None
                                kwargs['return_dict'] = False
                            else:
                                kwargs['past_key_values'] = None

                            out = self.m.decoder(**kwargs)
                            if isinstance(out, tuple):
                                return out[0]
                            elif hasattr(out, 'last_hidden_state'):
                                return out.last_hidden_state
                            return out[0]
                    # model.decoder.layers = model.decoder.layers[:1] # REMOVED layer limit
                    wrapped_model = AcestepWrapper(model).eval()
                    if "STATIC_SEQ_LEN" in os.environ:
                        del os.environ["STATIC_SEQ_LEN"]
                    
                    # Pre-compute sliding window mask for tracing (downseq=64 for seq=128)
                    trace_downseq = 64
                    sw = getattr(model.config, 'sliding_window', 128) or 128
                    sw_mask_np = np.zeros((1, 1, trace_downseq, trace_downseq), dtype=np.float32)
                    for i in range(trace_downseq):
                        for j in range(trace_downseq):
                            if abs(i - j) <= sw:
                                sw_mask_np[0, 0, i, j] = 0.0
                            else:
                                sw_mask_np[0, 0, i, j] = -65500.0
                    sw_mask_trace = torch.from_numpy(sw_mask_np)
                    
                    example_input = (
                        torch.randn((1, 128, 64), dtype=torch.float32), torch.randn((1,), dtype=torch.float32),
                        torch.randn((1,), dtype=torch.float32), torch.ones((1, 128), dtype=torch.float32),
                        torch.randn((1, 100, 2048), dtype=torch.float32), torch.randn((1, 100), dtype=torch.float32),
                        torch.randn((1, 128, 128), dtype=torch.float32), 
                        torch.arange(64, dtype=torch.int64).unsqueeze(0),
                        torch.arange(64, dtype=torch.int64),
                        sw_mask_trace
                    )
                    # Note: downseq = seq/2 due to patch_size=2
                    inputs_schema = [
                        ct.TensorType(name="hidden_states", shape=(1, ct.RangeDim(16, 32768, default=128), 64), dtype=np.float32),
                        ct.TensorType(name="timestep", shape=(1,), dtype=np.float32),
                        ct.TensorType(name="timestep_r", shape=(1,), dtype=np.float32),
                        ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(16, 32768, default=128)), dtype=np.float32),
                        ct.TensorType(name="encoder_hidden_states", shape=(1, ct.RangeDim(1, 4096, default=100), 2048), dtype=np.float32),
                        ct.TensorType(name="encoder_attention_mask", shape=(1, ct.RangeDim(1, 4096, default=100)), dtype=np.float32),
                        ct.TensorType(name="context_latents", shape=(1, ct.RangeDim(16, 32768, default=128), 128), dtype=np.float32),
                        ct.TensorType(name="position_ids", shape=(1, ct.RangeDim(8, 16384, default=64)), dtype=np.int32),
                        ct.TensorType(name="cache_position", shape=(ct.RangeDim(8, 16384, default=64),), dtype=np.int32),
                        ct.TensorType(name="sliding_window_mask", shape=(1, 1, ct.RangeDim(8, 16384, default=64), ct.RangeDim(8, 16384, default=64)), dtype=np.float32),
                    ]
                    outputs_schema = [ct.TensorType(name="velocity")]
                else:
                    is_text_encoder = "embedding" in item.name.lower() or "qwen" in item.name.lower()
                    class CausalMWrapper(torch.nn.Module):
                        def __init__(self, m, is_text_encoder):
                            super().__init__()
                            self.m = m
                            self.is_text_encoder = is_text_encoder
                        def forward(self, input_ids, attention_mask):
                            if self.is_text_encoder:
                                # For text encoders, we want the hidden states (dim 1024), not the vocab shape (dim 151669)
                                if hasattr(self.m, "model"):
                                    # Calling the inner model directly avoids tracing the LM head completely
                                    out = self.m.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                                    if isinstance(out, tuple):
                                        return out[0]
                                    return out
                                else:
                                    out = self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=False)
                                    if hasattr(out, "last_hidden_state"):
                                        return out.last_hidden_state
                                    if isinstance(out, tuple):
                                        return out[0]
                                    return out
                            else:
                                out = self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=False)
                                if isinstance(out, tuple):
                                    return out[0]
                                return out
                    wrapped_model = CausalMWrapper(model, is_text_encoder).eval()
                    os.environ["STATIC_SEQ_LEN"] = "128"
                    example_input = (torch.zeros((1, 128), dtype=torch.int32), torch.ones((1, 128), dtype=torch.int32))
                    inputs_schema = [
                        ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(16, 4096, default=128)), dtype=np.int32),
                        ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(16, 4096, default=128)), dtype=np.int32)
                    ]
                    outputs_schema = [ct.TensorType(name="hidden_states" if is_text_encoder else "logits")]

            log("  Tracing PyTorch model...")
            traced_model = trace_with_stack(wrapped_model, example_input, strict=False)

            log("  Converting to uncompressed Core ML representation (in-memory)...")
            mlmodel = ct.convert(
                traced_model,
                inputs=inputs_schema,
                outputs=outputs_schema,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.macOS14,
                compute_units=ct.ComputeUnit.CPU_ONLY,
                skip_model_load=True,
            )
            log("  [Done] Core ML conversion successful.")
            
            # Extract null condition embedding for DiT models before deleting
            null_emb = None
            if 'model' in locals() and hasattr(model, 'null_condition_emb'):
                null_emb = model.null_condition_emb.detach().cpu()
            
            # Extract encoder.* weights for ConditionEncoder (lyric_encoder, timbre_encoder, text_projector)
            # Without these weights, the ConditionEncoder runs with random initialization → garbled audio.
            encoder_tensors = None
            if 'model' in locals() and hasattr(model, 'encoder'):
                encoder_tensors = {}
                for k, v in model.state_dict().items():
                    if k.startswith('encoder.'):
                        encoder_tensors[k] = v.detach().cpu()
                if encoder_tensors:
                    log(f"  Extracted {len(encoder_tensors)} encoder weight keys for ConditionEncoder deployment.")
                else:
                    encoder_tensors = None
            
            # Collect tokenizer files for text encoder models
            is_text_encoder_model = "embedding" in item.name.lower() or "qwen" in item.name.lower()
            tokenizer_files_to_copy = []
            if is_text_encoder_model:
                for tok_file in ["tokenizer.json", "tokenizer_config.json"]:
                    src = item / tok_file
                    if src.exists():
                        tokenizer_files_to_copy.append(src)
                if tokenizer_files_to_copy:
                    log(f"  Found {len(tokenizer_files_to_copy)} tokenizer file(s) to deploy with text encoder.")

            # Extract embed_tokens.weight for text encoder models (used for lyric embedding direct lookup).
            # Without this, the lyric encoder receives full transformer hidden states (std~3.2)
            # instead of raw token embeddings (std~0.03), causing a 105x magnitude error.
            embed_tokens_tensor = None
            if is_text_encoder_model and 'model' in locals():
                embed_layer = getattr(model, 'embed_tokens', None)
                if embed_layer is None and hasattr(model, 'model'):
                    embed_layer = getattr(model.model, 'embed_tokens', None)
                if embed_layer is not None and hasattr(embed_layer, 'weight'):
                    embed_tokens_tensor = embed_layer.weight.detach().cpu().to(torch.float16)
                    log(f"  Extracted embed_tokens.weight: shape={list(embed_tokens_tensor.shape)}")
                else:
                    log("  Warning: Could not find embed_tokens.weight in text encoder model.")

            # Extract silence_latent for DiT models (used for context latents in text2music)
            silence_latent_tensor = None
            if 'model' in locals() and hasattr(model, 'silence_latent'):
                silence_latent_tensor = model.silence_latent.detach().cpu()
                log(f"  Extracted silence_latent: shape={list(silence_latent_tensor.shape)}")
            
            # Free memory
            if 'model' in locals(): del model
            if 'wrapped_model' in locals(): del wrapped_model
            if 'traced_model' in locals(): del traced_model
            gc.collect()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            for bits in bits_to_process:
                output_path = quantized_dir / f"{item.name}-coreml-{bits}bit.mlpackage"
                try:
                    if bits == 16:
                        log(f"  [16-bit] Saving uncompressed mlprogram directly to '{output_path}'...")
                        mlmodel.save(str(output_path))
                        log(f"  [16-bit] Successfully created '{output_path}'.")
                    else:
                        log(f"  [{bits}-bit] Applying palettization...")
                        op_config = ct.optimize.coreml.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=4096)
                        config = ct.optimize.coreml.OptimizationConfig(
                            global_config=op_config,
                            op_type_configs={"conv": None, "conv_transpose": None, "embedding": None}
                        )
                        compressed_mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config=config)
                        log(f"  [{bits}-bit] Saving to '{output_path}'...")
                        compressed_mlmodel.save(str(output_path))
                        log(f"  [{bits}-bit] Successfully created '{output_path}'.")
                        
                        if 'compressed_mlmodel' in locals(): del compressed_mlmodel
                        compressed_mlmodel = None
                        gc.collect()
                    log(f"  [{bits}-bit] Successfully created '{output_path}'.")
                    if null_emb is not None:
                        import safetensors.torch
                        null_path = output_path / "null_condition_embedding.safetensors"
                        safetensors.torch.save_file({"null_condition_emb": null_emb}, null_path)
                        log(f"  [{bits}-bit] Saved null_condition_embedding.safetensors to '{output_path}'.")
                    if encoder_tensors is not None:
                        import safetensors.torch
                        enc_path = output_path / "encoder.safetensors"
                        safetensors.torch.save_file(encoder_tensors, str(enc_path))
                        log(f"  [{bits}-bit] Saved encoder.safetensors ({len(encoder_tensors)} keys) to '{output_path}'.")
                    for tok_src in tokenizer_files_to_copy:
                        tok_dst = output_path / tok_src.name
                        shutil.copy(tok_src, tok_dst)
                        log(f"  [{bits}-bit] Copied {tok_src.name} to '{output_path}'.")
                    if embed_tokens_tensor is not None:
                        import safetensors.torch
                        et_path = output_path / "embed_tokens.safetensors"
                        safetensors.torch.save_file({"embed_tokens.weight": embed_tokens_tensor}, str(et_path))
                        log(f"  [{bits}-bit] Saved embed_tokens.safetensors to '{output_path}'.")
                    if silence_latent_tensor is not None:
                        import safetensors.torch
                        sl_path = output_path / "silence_latent.safetensors"
                        safetensors.torch.save_file({"latent": silence_latent_tensor}, str(sl_path))
                        log(f"  [{bits}-bit] Saved silence_latent.safetensors to '{output_path}'.")
                except Exception as e:
                    log(f"  [{bits}-bit] Error during compression: {e}")
                finally:
                    if 'compressed_mlmodel' in locals(): del compressed_mlmodel
                    compressed_mlmodel = None
                    gc.collect()
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()

        except Exception as e:
            log(f"  [Error] Failed to process model '{item.name}': {e}")
            traceback.print_exc()
        finally:
            if 'mlmodel' in locals(): del mlmodel
            gc.collect()

if __name__ == "__main__":
    main()
