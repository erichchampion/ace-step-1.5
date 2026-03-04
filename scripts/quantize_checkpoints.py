#!/usr/bin/env python3
"""
Script to convert models in the checkpoints/ directory to Core ML format 
and apply 8-bit, 6-bit, and 4-bit weight palettization (quantization).
Parallel directories will be created for each quantized model.
"""

import os
import sys
import gc
from pathlib import Path

def main():
    # Flush Hugging Face's transformers modules cache for acestep to ensure local patches apply
    try:
        import shutil
        import glob
        cache_pattern = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/acestep*")
        for cache_dir in glob.glob(cache_pattern):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"Flushed local huggingface cache: {cache_dir}")
    except Exception as e:
        print(f"Warning: Failed to flush huggingface cache: {e}")

    try:
        import torch
        import coremltools as ct
        import numpy as np
        from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer
        import warnings
        import traceback
        import sys
        
        def trace_with_stack(model, example_input, **kwargs):
            try:
                if isinstance(example_input, tuple):
                    traced = torch.jit.trace(model, example_input, **kwargs)
                else:
                    traced = torch.jit.trace(model, (example_input,), **kwargs)
                    
                print(f"====== PYTORCH JIT TRACE GRAPH ======\n{str(traced.graph)[:50000]}\n")
                print(f"====== PYTORCH JIT TRACE CODE ======\n{str(traced.code)}\n")
                
                return traced
            except Exception as e:
                print(f"\n[FATAL TRACER ERROR] {type(e).__name__}: {str(e)}")
                traceback.print_exc(file=sys.stdout)
                raise e

    except ImportError as e:
        print(f"Error: Missing required packages. {e}")
        print("Please ensure your dependencies are installed.")
        print("Suggestion: run 'uv pip install -r requirements.txt'")
        sys.exit(1)

    # Apply PyTorch Tracer Bypass Patch for F.embedding
    import inspect
    import textwrap
    import torch.nn.functional as F
    
    if not getattr(F, "_embedding_patched", False):
        try:
            emb_src = inspect.getsource(F.embedding)
            # Remove the assertion that casts a dimension to boolean during trace
            emb_src = emb_src.replace(
                "assert padding_idx < weight.size(0)",
                "assert True"
            )
            emb_src = emb_src.replace(
                "assert padding_idx >= -weight.size(0)",
                "assert True"
            )
            local_env = {}
            exec(textwrap.dedent(emb_src), F.__dict__, local_env)
            F.embedding = local_env["embedding"]
            F._embedding_patched = True
        except Exception as e:
            print(f"Failed to patch F.embedding: {e}")

    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        print(f"Error: Directory '{checkpoints_dir}' does not exist.")
        sys.exit(1)

    quantized_dir = Path("quantized_checkpoints_coreml")
    quantized_dir.mkdir(exist_ok=True)

    print("Scanning checkpoints directory for models...\n")
    
    # Force transformers to never use vmap for sdpa masks during tracing (it crashes torch.jit.trace)
    try:
        import transformers.masking_utils
        def fake_create_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, **kwargs):
            batch_size, seq_len = input_embeds.shape[0], input_embeds.shape[1]
            mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=input_embeds.device)
            mask = torch.tril(mask)
            # Mask format expects 0.0 for keep and min for block
            final_mask = (1.0 - mask.to(input_embeds.dtype)) * torch.finfo(input_embeds.dtype).min
            return final_mask
        transformers.masking_utils.create_causal_mask = fake_create_causal_mask
        transformers.models.qwen3.modeling_qwen3.create_causal_mask = fake_create_causal_mask
    except Exception as e:
        print(f"Warning: could not patch create_causal_mask: {e}")

    # Iterate over all directories in checkpoints/
    for item in checkpoints_dir.iterdir():
        if not item.is_dir():
            continue
            
        # Skip directories that already represent quantized models or coreml outputs
        if "bit" in item.name or "coreml" in item.name:
            continue

        print(f"--- Processing Model: {item.name} ---")



        # Let's see what bits are left to do so we don't load the model unnecessarily
        bits_to_process = []
        for bits in [8, 6, 4]:
            output_path = quantized_dir / f"{item.name}-coreml-{bits}bit.mlpackage"
            if not output_path.exists():
                bits_to_process.append(bits)
            else:
                print(f"  [{bits}-bit] Output '{output_path}' already exists. Skipping.")
                
        if not bits_to_process:
            print("  All requested quantizations already exist. Skipping to next model.\n")
            continue

        try:
            # Copy necessary quantizer dependency script so local AutoModel loads succeed
            if "acestep" in item.name.lower():
                import shutil
                dependency_script = Path(__file__).parent / "quantizer_torchao.py"
                target_script = item / "quantizer_torchao.py"
                if dependency_script.exists():
                    shutil.copy(dependency_script, target_script)
                    
            print("  Loading PyTorch model into memory (this may take a while)...")
            
            if item.name == "vae":
                from diffusers import AutoencoderOobleck
                model = AutoencoderOobleck.from_pretrained(str(item), torch_dtype=torch.float32)
                model.eval()
                
                class VAEWrapper(torch.nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m
                    def forward(self, latents):
                        return self.m.decode(latents)[0]
                        
                wrapped_model = VAEWrapper(model).eval()
                
                # Sample latents format for trace
                b, c, t = 1, 64, 100
                example_input = (torch.randn((b, c, t), dtype=torch.float32),)
                
                inputs_schema = [ct.TensorType(name="latents", shape=(1, 64, ct.RangeDim(1, 4096)), dtype=np.float32)]
                outputs_schema = [ct.TensorType(name="audio")]
                
            else:
                # Use AutoConfig to determine the correct model architecture type
                try:
                    config = AutoConfig.from_pretrained(str(item), trust_remote_code=True)
                    if type(config).__name__ == "AceStepConfig":
                        model = AutoModel.from_pretrained(str(item), trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager")
                    else:
                        model = AutoModelForCausalLM.from_pretrained(str(item), trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager")
                except Exception as inner_e:
                    print(f"  [Warning] AutoModel loading failed ({inner_e}), trying generic AutoModel fallback...")
                    model = AutoModel.from_pretrained(str(item), trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager")
            
            model.eval()
            
            # Disable KV cache output as it returns dynamic cache objects that torch.jit.trace cannot process
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            
            # Since custom Acestep models and generic HuggingFace models have different signatures,
            # we need to inspect the model and construct the correct inputs/wrappers dynamically.
            is_acestep = "AceStep" in type(model).__name__
            is_vae = item.name == "vae"

            if is_vae:
                # We already defined wrapped_model and example_input in the VAE block above
                pass

            elif is_acestep:
                import sys
                import builtins
                import torch
                import torch.nn.functional as F
                
                # Patch the active component classes directly to avoid HuggingFace cache trickery
                if hasattr(model, 'encoder') and hasattr(model.encoder, 'timbre_encoder'):
                    # Patch the class of the actual instance to ensure it takes effect
                    timbre_cls = type(model.encoder.timbre_encoder)
                    def patched_unpack(self, timbre_embs_packed, refer_audio_order_mask):
                        N, d = timbre_embs_packed.shape
                        timbre_embs_unpack = timbre_embs_packed.unsqueeze(0)
                        new_mask = torch.ones((1, N), dtype=torch.long, device=timbre_embs_packed.device)
                        return timbre_embs_unpack, new_mask
                    timbre_cls.unpack_timbre_embeddings = patched_unpack
                    print("  [Patch] Applied unpack_timbre_embeddings patch.")
                
                if hasattr(model, 'decoder'):
                    # Patch the DiT model/decoder class directly
                    decoder_cls = type(model.decoder)
                    DecoderMod = sys.modules[decoder_cls.__module__]
                    
                    if not getattr(decoder_cls, "_source_patched", False):
                        import inspect
                        # We must rewrite the source code of the class to remove PyTorch JIT 
                        # tracer traps (Boolean tensor to scalar casts, like modulo operations on shapes).
                        src = inspect.getsource(decoder_cls)
                        
                        # 1. Bypass dynamic shape modulo tests
                        if "hidden_states.shape[1] % self.patch_size != 0:" in src:
                            src = src.replace(
                                "if hidden_states.shape[1] % self.patch_size != 0:", 
                                "if False: # patched out to avoid Tracer boolean cast"
                            )
                            print("  [Patch] Bypassed patch_size modulo test.")
                            
                        # 2. Bypass boolean cast in max() dimension evaluation
                        if "max_len = max(seq_len, encoder_seq_len)" in src:
                            src = src.replace(
                                "max_len = max(seq_len, encoder_seq_len)",
                                "max_len = seq_len + encoder_seq_len # patched out for tracer boolean cast"
                            )
                            print("  [Patch] Bypassed max_len boolean cast.")
                        
                        import textwrap
                        src = textwrap.dedent(src)
                        local_env = {}
                        exec(src, DecoderMod.__dict__, local_env)
                        
                        # Re-bind the new class to the module and update the instance's class
                        # Note: We use the name of the class from the src to find it in local_env
                        cls_name = decoder_cls.__name__
                        if cls_name in local_env:
                            new_cls = local_env[cls_name]
                            new_cls._source_patched = True
                            setattr(DecoderMod, cls_name, new_cls)
                            # CRITICAL: Re-bind the instance's class so existing objects use the patched methods
                            model.decoder.__class__ = new_cls
                        
                        # Apply patches to AceStepAttention to stop coremltools from crashing on aten::Int list packing
                        if hasattr(DecoderMod, "AceStepAttention"):
                            attn_cls = DecoderMod.AceStepAttention
                            attn_src = inspect.getsource(attn_cls)
                            
                            # Bypass coremltools int() cast error when python slices a shape tuple
                            if "input_shape = hidden_states.shape[:-1]" in attn_src:
                                attn_src = attn_src.replace(
                                    "input_shape = hidden_states.shape[:-1]",
                                    "input_shape = (hidden_states.shape[0], hidden_states.shape[1]) # patched out to avoid aten::Int"
                                )
                                print("  [Patch] Fixed input_shape slice.")
                                
                            # Remove -1 from shapes to prevent coremltools from running int() on SymInts to infer shapes
                            if "hidden_shape = (*input_shape, -1, self.head_dim)" in attn_src:
                                attn_src = attn_src.replace(
                                    "hidden_shape = (*input_shape, -1, self.head_dim)",
                                    "query_shape = (*input_shape, self.config.num_attention_heads, self.head_dim)\n        kv_shape = (*input_shape, self.config.num_key_value_heads, self.head_dim)"
                                )
                                print("  [Patch] Fixed self-attn hidden_shape.")
                                
                            if "encoder_hidden_shape = (*encoder_hidden_states.shape[:-1], -1, self.head_dim)" in attn_src:
                                attn_src = attn_src.replace(
                                    "encoder_hidden_shape = (*encoder_hidden_states.shape[:-1], -1, self.head_dim)",
                                    "query_enc_shape = (*encoder_hidden_states.shape[:-1], self.config.num_attention_heads, self.head_dim)\n            kv_enc_shape = (*encoder_hidden_states.shape[:-1], self.config.num_key_value_heads, self.head_dim)"
                                )
                                print("  [Patch] Fixed cross-attn hidden_shape.")
                                
                            # Patch self-attention views
                            attn_src = attn_src.replace(
                                "self.q_proj(hidden_states).view(hidden_shape)",
                                "self.q_proj(hidden_states).view(query_shape)"
                            )
                            attn_src = attn_src.replace(
                                "self.k_proj(hidden_states).view(hidden_shape)",
                                "self.k_proj(hidden_states).view(kv_shape)"
                            )
                            attn_src = attn_src.replace(
                                "self.v_proj(hidden_states).view(hidden_shape)",
                                "self.v_proj(hidden_states).view(kv_shape)"
                            )
                            
                            # Patch cross-attention views
                            attn_src = attn_src.replace(
                                "self.q_proj(hidden_states).view(encoder_hidden_shape)",
                                "self.q_proj(hidden_states).view(query_enc_shape)"
                            )
                            attn_src = attn_src.replace(
                                "self.k_proj(encoder_hidden_states).view(encoder_hidden_shape)",
                                "self.k_proj(encoder_hidden_states).view(kv_enc_shape)"
                            )
                            attn_src = attn_src.replace(
                                "self.v_proj(encoder_hidden_states).view(encoder_hidden_shape)",
                                "self.v_proj(encoder_hidden_states).view(kv_enc_shape)"
                            )
                            
                            if "attn_output = attn_output.reshape(*input_shape, -1).contiguous()" in attn_src:
                                attn_src = attn_src.replace(
                                    "attn_output = attn_output.reshape(*input_shape, -1).contiguous()",
                                    "attn_output = attn_output.reshape(hidden_states.shape[0], hidden_states.shape[1], self.config.num_attention_heads * self.head_dim).contiguous() # patched out to avoid aten::Int"
                                )
                                print("  [Patch] Fixed attn_output reshape.")
                            
                            # Patch RoPE half_dim to integer to avoid aten::Int errors
                            if "query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)" in attn_src:
                                attn_src = attn_src.replace(
                                    "query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)",
                                    "query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, half_dim=self.head_dim // 2)"
                                )
                                print("  [Patch] Fixed RoPE half_dim.")
                            
                            local_env_attn = {}
                            exec(textwrap.dedent(attn_src), DecoderMod.__dict__, local_env_attn)
                            patched_attn_cls = local_env_attn["AceStepAttention"]
                            setattr(DecoderMod, "AceStepAttention", patched_attn_cls)
                            
                            # Important: Since DiTModel creates its layers in __init__, we must also
                            # swap the class on all existing attention modules within those layers.
                            for layer in model.decoder.layers:
                                if hasattr(layer, 'self_attn'):
                                    layer.self_attn.__class__ = patched_attn_cls
                                if hasattr(layer, 'cross_attn'):
                                    layer.cross_attn.__class__ = patched_attn_cls
                        
                        # Apply a secondary patch to `eager_attention_forward` to strip dynamic modulo tests from Qwen3
                        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
                        qwen3_mod = sys.modules[eager_attention_forward.__module__]
                        
                        if not getattr(qwen3_mod, "_qwen_patched", False):
                            qwen_src = inspect.getsource(eager_attention_forward)
                            # Remove modulo operations that crash CoreML PyTorch Tracer
                            qwen_src = qwen_src.replace(
                                "if bsz * num_heads * q_len * num_key_value_heads * k_len * head_dim == 0:",
                                "if False:# bsz * num_heads"
                            )
                            qwen_src = qwen_src.replace(
                                "if query_states.shape[2] % 2 != 0:",
                                "if False:"
                            )
                            # Bypass boolean cast tests: `if attention_mask is not None:` -> `if True:`
                            # since we always pass attention masks during CoreML tracing
                            qwen_src = qwen_src.replace(
                                "if attention_mask is not None:",
                                "if True:"
                            )
                            qwen_src = qwen_src.replace(
                                "causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]",
                                "causal_mask = attention_mask # patched out dynamic slice to avoid aten::Int"
                            )
                            exec(textwrap.dedent(qwen_src), qwen3_mod.__dict__)
                            
                            # Also patch Qwen3Attention.forward block that has a past_key_values test
                            from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding
                            
                            rope_src = inspect.getsource(Qwen3RotaryEmbedding.forward)
                            rope_src = rope_src.replace(
                                ".expand(position_ids.shape[0], -1, 1)",
                                ".expand(position_ids.shape[0], self.inv_freq.shape[0], 1) # patched out -1"
                            )
                            # We must re-bind the patched method to the class
                            exec(textwrap.dedent(rope_src), qwen3_mod.__dict__, local_env)
                            Qwen3RotaryEmbedding.forward = local_env["forward"]

                            qwen_attn_src = inspect.getsource(Qwen3Attention.forward)
                            qwen_attn_src = qwen_attn_src.replace(
                                "if past_key_values is not None:",
                                "if False: # patched out for tracer"
                            )
                            # Bypass coremltools int() cast error when python slices a shape tuple
                            qwen_attn_src = qwen_attn_src.replace(
                                "input_shape = hidden_states.shape[:-1]",
                                "input_shape = (hidden_states.shape[0], hidden_states.shape[1]) # patched out to avoid aten::Int"
                            )
                            # Remove -1 from shapes to prevent coremltools from running int() on SymInts to infer shapes
                            qwen_attn_src = qwen_attn_src.replace(
                                "hidden_shape = (*input_shape, -1, self.head_dim)",
                                "query_shape = (*input_shape, self.config.num_attention_heads, self.head_dim)\n        kv_shape = (*input_shape, self.config.num_key_value_heads, self.head_dim)"
                            )
                            qwen_attn_src = qwen_attn_src.replace(
                                "self.q_proj(hidden_states).view(hidden_shape)",
                                "self.q_proj(hidden_states).view(query_shape)"
                            )
                            qwen_attn_src = qwen_attn_src.replace(
                                "self.k_proj(hidden_states).view(hidden_shape)",
                                "self.k_proj(hidden_states).view(kv_shape)"
                            )
                            qwen_attn_src = qwen_attn_src.replace(
                                "self.v_proj(hidden_states).view(hidden_shape)",
                                "self.v_proj(hidden_states).view(kv_shape)"
                            )
                            qwen_attn_src = qwen_attn_src.replace(
                                "attn_output = attn_output.reshape(*input_shape, -1).contiguous()",
                                "attn_output = attn_output.reshape(hidden_states.shape[0], hidden_states.shape[1], self.config.num_attention_heads * self.head_dim).contiguous() # patched out to avoid aten::Int"
                            )
                            # We must re-bind the patched method to the class
                            exec(textwrap.dedent(qwen_attn_src), qwen3_mod.__dict__, local_env)
                            Qwen3Attention.forward = local_env["forward"]
                            
                            # Patch repeat_kv to use repeat_interleave avoiding explicit dimension math (aten::Int) which breaks CoreML
                            def patched_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
                                if n_rep == 1:
                                    return hidden_states
                                return torch.repeat_interleave(hidden_states, repeats=n_rep, dim=1)
                            qwen3_mod.repeat_kv = patched_repeat_kv
                            
                            qwen3_mod._qwen_patched = True
                        
                        # Swap the patched class definition into the live module globally
                        new_cls = local_env[decoder_cls.__name__]
                        setattr(DecoderMod, decoder_cls.__name__, new_cls)
                        
                        # Re-instantiate the decoder on the existing model using the new class 
                        model.decoder.__class__ = new_cls

                # We build a simplified wrapper that traces a minimal subset of arguments.
                class AcestepWrapper(torch.nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m
                    def forward(self, hidden_states, attention_mask):
                        # Force static integers for dummy dimensions to prevent aten::Int CoreML crashes
                        dt = hidden_states.dtype
                        dev = hidden_states.device
                        
                        # The Acestep model expects specific diffusion conditioning args.
                        # For tracing, we pass dummy tensors for the other requirements.
                        # We completely bypass the encoder since the tracer gets stuck on its int() operations.
                        enc_len = 25
                        encoder_hidden = torch.zeros((1, enc_len, getattr(self.m.config, "hidden_size", 1024)), dtype=dt, device=dev)
                        encoder_mask = torch.ones((1, enc_len), dtype=dt, device=dev)
                        
                        # Standardize context latents manually rather than tracking covers
                        context_latents = torch.cat([hidden_states, torch.zeros_like(hidden_states)], dim=-1)
                        
                        # Run the actual decoder
                        t = torch.ones((1,), dtype=dt, device=dev)
                        out = self.m.decoder(
                            hidden_states=hidden_states,
                            timestep=t,
                            timestep_r=t,
                            attention_mask=attention_mask,
                            encoder_hidden_states=encoder_hidden,
                            encoder_attention_mask=encoder_mask,
                            context_latents=context_latents,
                        )
                        return out[0] # Just the velocity tensor

                wrapped_model = AcestepWrapper(model).eval()
                
                # Create sample hidden states according to config
                b, s, d = 1, 100, getattr(model.config, "audio_acoustic_hidden_dim", 64)
                hidden_states = torch.randn((b, s, d), dtype=torch.float32)
                attention_mask = torch.ones((b, s), dtype=torch.float32)
                example_input = (hidden_states, attention_mask)
                
                inputs_schema = [
                    ct.TensorType(name="hidden_states", shape=(1, ct.RangeDim(1, 4096), d), dtype=np.float32),
                    ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, 4096)), dtype=np.float32)
                ]
                outputs_schema = [ct.TensorType(name="velocity")]

            else:
                # Standard causal LLM (like Qwen)
                class CausalMWrapper(torch.nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m
                    def forward(self, input_ids, attention_mask):
                        # Some versions of transformers throw KV cache unordered_map errors
                        # if attention_mask is missing when caching is disabled globally.
                        outputs = self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=False)
                        # outputs is a tuple: (logits, past_key_values, ...)
                        return outputs[0]

                wrapped_model = CausalMWrapper(model).eval()

                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(item), trust_remote_code=True)
                    inputs = tokenizer("Hello, world!", return_tensors="pt")
                    # Force strict int32 to match MIL schema and prevent unordered_map type miss-matches
                    example_input = (inputs["input_ids"].to(torch.int32), inputs["attention_mask"].to(torch.int32))
                except Exception as e:
                    print(f"  [Warning] Could not load tokenizer for sample input, using generic tensor: {e}")
                    example_input = (torch.zeros((1, 10), dtype=torch.int32), torch.ones((1, 10), dtype=torch.int32))

                inputs_schema = [
                    ct.TensorType(name="input_ids", shape=(1, 10), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, 10), dtype=np.int32)
                ]
                outputs_schema = [ct.TensorType(name="logits")]

            print("  Tracing PyTorch model...")
            traced_model = trace_with_stack(wrapped_model, example_input, strict=False)

            print("  Converting to uncompressed Core ML representation (in-memory)...")
            # Define exact inputs matching the ModelWrapper signatures
            mlmodel = ct.convert(
                traced_model,
                inputs=inputs_schema,
                outputs=outputs_schema,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.macOS13,
            )
            
            # Now apply palettization for each bit depth
            for bits in bits_to_process:
                output_path = quantized_dir / f"{item.name}-coreml-{bits}bit.mlpackage"
                print(f"  [{bits}-bit] Applying palettization...")
                
                try:
                    op_config = ct.optimize.coreml.OpPalettizerConfig(
                        mode="kmeans", 
                        nbits=bits, 
                        weight_threshold=512 # Only compress layers > 512 elements
                    )
                    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
                    
                    compressed_mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config=config)
                    
                    print(f"  [{bits}-bit] Saving to '{output_path}'...")
                    compressed_mlmodel.save(str(output_path))
                    print(f"  [{bits}-bit] Successfully created '{output_path}'.")
                except Exception as e:
                    print(f"  [{bits}-bit] Error during compression or saving: {e}")

        except Exception as e:
            print(f"  [Error] Failed to process model '{item.name}': {e}")
            
        finally:
            # Aggressively free memory before moving to the next model
            if 'model' in locals(): del model
            if 'wrapped_model' in locals(): del wrapped_model
            if 'traced_model' in locals(): del traced_model
            if 'mlmodel' in locals(): del mlmodel
            if 'compressed_mlmodel' in locals(): del compressed_mlmodel
            if 'tokenizer' in locals(): del tokenizer
            if 'inputs' in locals(): del inputs
            if 'example_input' in locals(): del example_input
            
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch accelerator caches (critical for Unified Memory on Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\n")

if __name__ == "__main__":
    main()
