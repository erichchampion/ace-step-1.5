"""Initialization-adjacent utility mixin for AceStepHandler."""

import os
from typing import List, Optional

import torch


class InitServiceMixin:
    def get_available_checkpoints(self) -> List[str]:
        """Return project root directory path"""
        # Get project root (handler.py is in acestep/, so go up two levels to project root)
        project_root = self._get_project_root()
        # default checkpoints
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        if os.path.exists(checkpoint_dir):
            return [checkpoint_dir]
        else:
            return []

    def get_available_acestep_v15_models(self) -> List[str]:
        """Scan and return all model directory names starting with 'acestep-v15-'"""
        # Get project root
        project_root = self._get_project_root()
        checkpoint_dir = os.path.join(project_root, "checkpoints")

        models = []
        if os.path.exists(checkpoint_dir):
            # Scan all directories starting with 'acestep-v15-' in checkpoints folder
            for item in os.listdir(checkpoint_dir):
                item_path = os.path.join(checkpoint_dir, item)
                if os.path.isdir(item_path) and item.startswith("acestep-v15-"):
                    models.append(item)

        # Sort by name
        models.sort()
        return models

    def is_flash_attention_available(self, device: Optional[str] = None) -> bool:
        """Check whether flash attention can be used on the target device."""
        target_device = str(device or self.device or "auto").split(":", 1)[0]
        if target_device == "auto":
            if not torch.cuda.is_available():
                return False
        elif target_device != "cuda":
            return False
        if not torch.cuda.is_available():
            return False
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def is_turbo_model(self) -> bool:
        """Check if the currently loaded model is a turbo model"""
        if self.config is None:
            return False
        return getattr(self.config, "is_turbo", False)

    def _empty_cache(self):
        """Clear accelerator memory cache (CUDA, XPU, or MPS)."""
        device_type = self.device if isinstance(self.device, str) else self.device.type
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _synchronize(self):
        """Synchronize accelerator operations (CUDA, XPU, or MPS)."""
        device_type = self.device if isinstance(self.device, str) else self.device.type
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif device_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()

    def _memory_allocated(self):
        """Get current accelerator memory usage in bytes, or 0 for unsupported backends."""
        device_type = self.device if isinstance(self.device, str) else self.device.type
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        # MPS and XPU don't expose per-tensor memory tracking
        return 0

    def _max_memory_allocated(self):
        """Get peak accelerator memory usage in bytes, or 0 for unsupported backends."""
        device_type = self.device if isinstance(self.device, str) else self.device.type
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
        return 0
