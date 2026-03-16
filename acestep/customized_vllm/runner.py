"""Model runner: loads model, manages KV cache, CUDA graphs, and runs forward passes."""

import torch
import sys

from acestep.customized_vllm.context import set_context, get_context, reset_context
from acestep.customized_vllm.model import Qwen3ForCausalLM
from acestep.customized_vllm.sampling import Sampler
from acestep.customized_vllm.loader import load_model
from acestep.customized_vllm.sequence import Sequence
from acestep.debug_utils import debug_start, debug_end


class ModelRunner:
    """Loads a Qwen3 model, allocates KV cache, captures CUDA graphs, runs inference."""

    def __init__(self, hf_config, model_path: str, block_size: int, max_num_seqs: int,
                 max_num_batched_tokens: int, max_model_len: int, gpu_memory_utilization: float,
                 enforce_eager: bool):
        torch._dynamo.config.capture_scalar_outputs = True
        self.block_size = block_size
        self.enforce_eager = enforce_eager
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.hf_config = hf_config

        torch.cuda.set_device(0)
        saved_dtype = torch.get_default_dtype()

        gpu_props = torch.cuda.get_device_properties(0)
        bf16_ok = (gpu_props.major, gpu_props.minor) >= (8, 0)
        raw = getattr(hf_config, "dtype", getattr(hf_config, "torch_dtype", None))
        if isinstance(raw, str):
            _map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
            raw = _map.get(raw.replace("torch.", ""), None)
        self.dtype = (raw if isinstance(raw, torch.dtype) and raw.is_floating_point else
                      torch.bfloat16 if bf16_ok else torch.float16)
        if self.dtype == torch.bfloat16 and not bf16_ok:
            self.dtype = torch.float16

        torch.set_default_dtype(self.dtype)
        torch.set_default_device("cuda")

        self.model = Qwen3ForCausalLM(hf_config)
        _t = debug_start("load_model", prefix="tensor.vllm")
        load_model(self.model, model_path)
        debug_end("load_model", _t, prefix="tensor.vllm")
        self.sampler = Sampler()

        self._alloc_buffers()
        self._warmup()
        self._alloc_kv_cache()
        if not enforce_eager:
            self._capture_cuda_graphs()

        torch.set_default_device("cpu")
        torch.set_default_dtype(saved_dtype)

    # -- Buffer pre-allocation --

    def _alloc_buffers(self):
        """Pre-allocate pinned CPU buffers for fast GPU transfer."""
        bs = self.max_num_seqs
        max_blocks = (self.max_model_len + self.block_size - 1) // self.block_size
        pin = dict(dtype=torch.float32, device="cpu", pin_memory=True)
        pin_i32 = dict(dtype=torch.int32, device="cpu", pin_memory=True)
        pin_i64 = dict(dtype=torch.int64, device="cpu", pin_memory=True)
        self._buf_temps = torch.zeros(bs, **pin)
        self._buf_cfg = torch.zeros(bs, **pin)
        self._buf_topk = torch.zeros(bs, **pin_i32)
        self._buf_topp = torch.zeros(bs, **pin)
        self._buf_rep = torch.zeros(bs, **pin)
        self._buf_ids = torch.zeros(bs, **pin_i64)
        self._buf_pos = torch.zeros(bs, **pin_i64)
        self._buf_slot = torch.zeros(bs, **pin_i32)
        self._buf_ctx = torch.zeros(bs, **pin_i32)

    # -- Warmup & KV cache --

    def _warmup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        n = min(self.max_num_batched_tokens // self.max_model_len, self.max_num_seqs)
        seqs = [Sequence([0] * self.max_model_len) for _ in range(n)]
        self._run_prefill(seqs)
        reset_context()
        torch.cuda.empty_cache()

    def _alloc_kv_cache(self):
        _t = debug_start("allocate_kv_cache", prefix="tensor.vllm")
        hf = self.hf_config
        free, total = torch.cuda.mem_get_info()
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        import os
        sim = os.environ.get("MAX_CUDA_VRAM")
        if sim:
            try:
                cap = float(sim) * 1024**3
                if cap < total:
                    total = int(cap)
                    free = max(0, total - torch.cuda.memory_reserved())
            except (ValueError, TypeError):
                pass

        num_kv_heads = hf.num_key_value_heads
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        block_bytes = 2 * hf.num_hidden_layers * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize

        target = total * self.gpu_memory_utilization
        avail = min(free * 0.9, target - current, max(0, free - 1024**3) * 0.9)
        if avail <= 0:
            avail = free * 0.5

        self.num_kvcache_blocks = max(1, int(avail) // block_bytes)
        cap = self.num_kvcache_blocks * self.block_size
        gb = self.num_kvcache_blocks * block_bytes / 1024**3
        print(f"[customized_vllm] KV cache: {self.num_kvcache_blocks} blocks, "
              f"{cap} tokens, {gb:.2f} GB")

        self.kv_cache = torch.empty(
            2, hf.num_hidden_layers, self.num_kvcache_blocks,
            self.block_size, num_kv_heads, head_dim,
        )
        layer_id = 0
        for m in self.model.modules():
            if hasattr(m, "k_cache") and hasattr(m, "v_cache"):
                m.k_cache = self.kv_cache[0, layer_id]
                m.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        debug_end("allocate_kv_cache", _t, prefix="tensor.vllm")

    # -- Input preparation --

    def _prepare_block_tables(self, seqs):
        max_len = max(len(s.block_table) for s in seqs)
        bt = [s.block_table + [-1] * (max_len - len(s.block_table)) for s in seqs]
        return torch.tensor(bt, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def _run_prefill(self, seqs):
        """Prepare prefill inputs, run model forward, return logits."""
        ids, pos, cu_q, cu_k = [], [], [0], [0]
        max_sq = max_sk = 0
        slot_map = []
        for seq in seqs:
            n = len(seq)
            ids.extend(seq.token_ids)
            pos.extend(range(n))
            cu_q.append(cu_q[-1] + n)
            cu_k.append(cu_k[-1] + n)
            max_sq = max(n, max_sq)
            max_sk = max(n, max_sk)
            for i in range(seq.num_blocks):
                if not seq.block_table:
                    continue
                start = seq.block_table[i] * self.block_size
                end = start + (seq.last_block_tokens if i == seq.num_blocks - 1 else self.block_size)
                slot_map.extend(range(start, end))

        ids = torch.tensor(ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        pos = torch.tensor(pos, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_q = torch.tensor(cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_k = torch.tensor(cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        sm = torch.tensor(slot_map, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_q, cu_k, max_sq, max_sk, sm, None, None)
        return self._run_model(ids, pos, is_prefill=True)

    def _run_decode(self, seqs):
        """Prepare decode inputs, run model forward, return logits."""
        bs = len(seqs)
        for i, s in enumerate(seqs):
            self._buf_ids[i] = s.last_token
            self._buf_pos[i] = len(s) - 1
            self._buf_ctx[i] = len(s)
            self._buf_slot[i] = s.block_table[-1] * self.block_size + s.last_block_tokens - 1

        ids = self._buf_ids[:bs].cuda(non_blocking=True)
        pos = self._buf_pos[:bs].cuda(non_blocking=True)
        sm = self._buf_slot[:bs].cuda(non_blocking=True)
        cl = self._buf_ctx[:bs].cuda(non_blocking=True)
        bt = self._prepare_block_tables(seqs)
        set_context(False, slot_mapping=sm, context_lens=cl, block_tables=bt)
        return self._run_model(ids, pos, is_prefill=False)

    # -- Model forward --

    @torch.inference_mode()
    def _run_model(self, input_ids, positions, is_prefill):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))

        bs = input_ids.size(0)
        ctx = get_context()
        gv = self.graph_vars
        max_cols = gv["block_tables"].size(1)
        if (ctx.block_tables.size(1) > max_cols or ctx.block_tables.size(0) != bs
                or ctx.slot_mapping.size(0) != bs or ctx.context_lens.size(0) != bs):
            return self.model.compute_logits(self.model(input_ids, positions))

        graph = self.graphs[next(x for x in self._graph_bs if x >= bs)]
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = ctx.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = ctx.context_lens
        gv["block_tables"][:bs].fill_(-1)
        gv["block_tables"][:bs, :ctx.block_tables.size(1)] = ctx.block_tables
        graph.replay()
        return self.model.compute_logits(gv["outputs"][:bs])

    def prepare_sample_params(self, seqs, is_cfg):
        """Pack per-sequence sampling params into GPU tensors."""
        targets = seqs[:len(seqs) // 2] if is_cfg else seqs
        n = len(targets)
        has_topk = has_topp = has_rep = False
        for i, s in enumerate(targets):
            self._buf_temps[i] = s.temperature
            self._buf_cfg[i] = s.cfg_scale
            self._buf_topk[i] = s.top_k if s.top_k else 0
            self._buf_topp[i] = s.top_p if s.top_p else 1.0
            self._buf_rep[i] = s.repetition_penalty if s.repetition_penalty else 1.0
            if s.top_k and s.top_k > 0:
                has_topk = True
            if s.top_p and s.top_p < 1.0:
                has_topp = True
            if s.repetition_penalty and s.repetition_penalty != 1.0:
                has_rep = True
        return (
            self._buf_temps[:n].cuda(non_blocking=True),
            self._buf_cfg[:n].cuda(non_blocking=True),
            self._buf_topk[:n].cuda(non_blocking=True) if has_topk else None,
            self._buf_topp[:n].cuda(non_blocking=True) if has_topp else None,
            self._buf_rep[:n].cuda(non_blocking=True) if has_rep else None,
        )

    def run(self, seqs, is_prefill):
        """Full forward + sampling step. Returns list of sampled token IDs."""
        is_cfg = seqs[0].cfg_scale > 1.0 and seqs[0].paired_seq is not None
        logits = (self._run_prefill(seqs) if is_prefill else self._run_decode(seqs))
        reset_context()
        temps, cfg_s, topk, topp, rep_pen = self.prepare_sample_params(seqs, is_cfg)

        if is_cfg:
            nc = len(seqs) // 2
            cond, uncond = logits[:nc], logits[nc:]
            cond = self._apply_rep_penalty(cond, seqs[:nc], rep_pen)
            cfg_logits = uncond + cfg_s.unsqueeze(1) * (cond - uncond)
            cfg_logits = self._apply_logits_processors(cfg_logits, seqs[:nc])
            tids = self.sampler(cfg_logits, temps, topk, topp).tolist()
            if seqs[0].logits_processor_update_state:
                seqs[0].logits_processor_update_state(tids[0])
            return tids

        logits = self._apply_rep_penalty(logits, seqs, rep_pen)
        logits = self._apply_logits_processors(logits.clone(), seqs)
        tids = self.sampler(logits, temps, topk, topp).tolist()
        if seqs and seqs[0].logits_processor_update_state:
            seqs[0].logits_processor_update_state(tids[0])
        return tids

    def _apply_rep_penalty(self, logits, seqs, penalties):
        if penalties is None:
            return logits
        for i, seq in enumerate(seqs):
            p = penalties[i].item()
            if p == 1.0:
                continue
            comp = torch.tensor(seq.completion_token_ids, device=logits.device)
            if len(comp) == 0:
                continue
            mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
            mask[comp] = True
            penalized = torch.where(logits[i] < 0, logits[i] * p, logits[i] / p)
            logits[i] = torch.where(mask, penalized, logits[i])
        return logits

    def _apply_logits_processors(self, logits, seqs):
        for i, seq in enumerate(seqs):
            if seq.logits_processor is not None:
                ids_t = torch.tensor([seq.token_ids], device=logits.device)
                processed = seq.logits_processor(ids_t, logits[i:i+1].clone())
                logits[i] = processed[0]
        return logits

    # -- CUDA graph capture --

    @torch.inference_mode()
    def _capture_cuda_graphs(self):
        _t = debug_start("capture_cudagraph", prefix="tensor.vllm")
        max_bs = min(self.max_num_seqs, 512)
        max_blocks = (self.max_model_len + self.block_size - 1) // self.block_size
        ids = torch.zeros(max_bs, dtype=torch.int64)
        pos = torch.zeros(max_bs, dtype=torch.int64)
        sm = torch.zeros(max_bs, dtype=torch.int32)
        cl = torch.zeros(max_bs, dtype=torch.int32)
        bt = torch.zeros(max_bs, max_blocks, dtype=torch.int32)
        out = torch.zeros(max_bs, self.hf_config.hidden_size)
        self._graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        pool = None
        for bs in reversed(self._graph_bs):
            g = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=sm[:bs], context_lens=cl[:bs], block_tables=bt[:bs])
            out[:bs] = self.model(ids[:bs], pos[:bs])
            with torch.cuda.graph(g, pool):
                out[:bs] = self.model(ids[:bs], pos[:bs])
            if pool is None:
                pool = g.pool()
            self.graphs[bs] = g
            torch.cuda.synchronize()
            reset_context()
        self.graph_vars = dict(input_ids=ids, positions=pos, slot_mapping=sm,
                               context_lens=cl, block_tables=bt, outputs=out)
        debug_end("capture_cudagraph", _t, prefix="tensor.vllm")

    def exit(self):
        if not self.enforce_eager:
            del self.graphs, self.graph_vars
        torch.cuda.synchronize()
