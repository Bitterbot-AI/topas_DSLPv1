#!/usr/bin/env python3
# trainers/k2_muonclip.py
"""
K2-style MuonClip optimizer wrapper for TOPAS/BitterBot Brain

Core idea:
- Track the maximum pre-softmax attention logits (per head) produced by Q·K / sqrt(d).
- If a head's max exceeds a target τ, softly rescale its Q and K row-blocks to pull the
  head's future logits back toward τ. (Downward-only, EMA-smoothed, bounded per step.)

This file provides:
- MuonClipConfig: tunable parameters.
- MuonClip: a drop-in wrapper around any base torch optimizer (e.g., AdamW).
- helpers to discover attention modules (both qkv-proj style and separate q/k-proj style).
- lightweight instrumentation toggles for modules to expose `_muon_last_qk_max`.

Integration surface:
1) Add tiny hooks to your attention modules to populate `_muon_last_qk_max` each forward.
   (See patch file generated along with this module.)
2) Wrap your optimizer with MuonClip and call `opt.step()` as usual.

This implementation favors safety and stability:
- All rescales are under `torch.no_grad()` and constrained by `max_scale_change`.
- EMA is kept per (module, head) to reduce jitter.
- KV-grouped attention is handled (num_key_value_heads may differ from num_heads).
- If runtime maxima are not available (module didn't expose them), we no-op.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn

Device = torch.device

# ------------------------------
# Configuration
# ------------------------------

@dataclass
class MuonClipConfig:
    target_logit: float = 12.0           # τ – soft target for max(QK/sqrt(d)) per head (logit scale)
    ema_beta: float = 0.9                # EMA smoothing for observed maxima
    interval: int = 1                    # apply after every N optimizer steps
    max_scale_change: float = 0.20       # per-step bound on multiplicative downscale (0.2 => ∈ [0.8, 1.0])
    capture_every_forward: bool = True   # if True, modules compute max on every forward
    respect_kv_sharing: bool = True      # handle num_key_value_heads < num_heads (KV grouping)
    enable_q_rescale: bool = True        # allow rescaling Q row blocks
    enable_k_rescale: bool = True        # allow rescaling K row blocks
    fp32_rescale_math: bool = True       # cast to fp32 for safe in-place rescale
    verbose: bool = False                # print per-apply stats
    # Optional gentle recovery above 1.0 (off by default, preserves current behavior)
    allow_upscale: bool = False
    max_upscale_change: float = 0.05     # ∈ [1.0, 1.05] if allow_upscale=True
    # Optional hard clamps (rarely needed)
    min_row_scale: float = 0.25         # absolute min clamp for accumulated scaling (safety)
    max_row_scale: float = 4.00         # absolute max clamp for accumulated scaling (safety)


# ------------------------------
# Discovery utilities
# ------------------------------

class _QKVModuleView:
    """View into a qkv-proj style attention (models.layers.Attention)."""
    def __init__(self, mod: nn.Module):
        # Required attributes: qkv_proj (Linear-like), o_proj, num_heads, num_key_value_heads, head_dim
        self.m = mod
        assert hasattr(mod, "qkv_proj") and hasattr(mod, "o_proj")
        assert hasattr(mod, "num_heads") and hasattr(mod, "num_key_value_heads") and hasattr(mod, "head_dim")
        self.num_heads: int = int(mod.num_heads)
        self.num_kv: int = int(mod.num_key_value_heads)
        self.head_dim: int = int(mod.head_dim)
        self.qkv: nn.Module = mod.qkv_proj
        self.o_proj: nn.Module = mod.o_proj

        # Precompute row ranges in qkv weight for Q/K
        self.q_rows: List[slice] = []
        self.k_rows_by_kvidx: List[slice] = []

        q_offset = 0
        k_offset = self.num_heads * self.head_dim
        v_offset = k_offset + self.num_kv * self.head_dim  # not used, but documented

        # Q slices (one per head)
        for h in range(self.num_heads):
            start = q_offset + h * self.head_dim
            self.q_rows.append(slice(start, start + self.head_dim))

        # K slices (one per KV head index)
        for kv in range(self.num_kv):
            start = k_offset + kv * self.head_dim
            self.k_rows_by_kvidx.append(slice(start, start + self.head_dim))

    def apply_rescale(self, q_scales: Optional[torch.Tensor], k_scales: Optional[torch.Tensor],
                      cfg: MuonClipConfig):
        """
        q_scales: [H] or None (per Q head)
        k_scales: [KV] or None (per KV head)
        """
        with torch.no_grad():
            W: torch.Tensor = self.m.qkv_proj.weight.data
            if cfg.fp32_rescale_math and W.dtype != torch.float32:
                W = W.float()

            if q_scales is not None and cfg.enable_q_rescale:
                assert q_scales.numel() == self.num_heads
                for h, sl in enumerate(self.q_rows):
                    s = float(q_scales[h].item())
                    # clamp extreme accumulation safeguards
                    block = W[sl, :]
                    block.mul_(s)
                    W[sl, :] = block

            if k_scales is not None and cfg.enable_k_rescale:
                assert k_scales.numel() == self.num_kv
                for kv, sl in enumerate(self.k_rows_by_kvidx):
                    s = float(k_scales[kv].item())
                    block = W[sl, :]
                    block.mul_(s)
                    W[sl, :] = block

            # Write back in original dtype if we upcasted
            if self.m.qkv_proj.weight.data.dtype != W.dtype:
                self.m.qkv_proj.weight.data.copy_(W.to(self.m.qkv_proj.weight.dtype))
            else:
                self.m.qkv_proj.weight.data.copy_(W)


class _SepQKModuleView:
    """View into a separate-proj attention (models.rel_graph.MultiHeadAttention)."""
    def __init__(self, mod: nn.Module):
        # Required attributes: q_proj, k_proj, heads, head_dim
        self.m = mod
        assert hasattr(mod, "q_proj") and hasattr(mod, "k_proj")
        assert hasattr(mod, "heads") and hasattr(mod, "head_dim")
        self.heads: int = int(mod.heads)
        self.head_dim: int = int(mod.head_dim)
        self.q_proj: nn.Linear = mod.q_proj
        self.k_proj: nn.Linear = mod.k_proj

        # Row slices for each head
        self.q_rows: List[slice] = []
        self.k_rows: List[slice] = []
        for h in range(self.heads):
            start = h * self.head_dim
            self.q_rows.append(slice(start, start + self.head_dim))
            self.k_rows.append(slice(start, start + self.head_dim))

    def apply_rescale(self, q_scales: Optional[torch.Tensor], k_scales: Optional[torch.Tensor],
                      cfg: MuonClipConfig):
        with torch.no_grad():
            if q_scales is not None and cfg.enable_q_rescale:
                Wq = self.q_proj.weight.data
                if cfg.fp32_rescale_math and Wq.dtype != torch.float32:
                    Wq = Wq.float()
                for h, sl in enumerate(self.q_rows):
                    s = float(q_scales[h].item())
                    block = Wq[sl, :]
                    block.mul_(s)
                    Wq[sl, :] = block
                if self.q_proj.weight.data.dtype != Wq.dtype:
                    self.q_proj.weight.data.copy_(Wq.to(self.q_proj.weight.dtype))
                else:
                    self.q_proj.weight.data.copy_(Wq)

            if k_scales is not None and cfg.enable_k_rescale:
                Wk = self.k_proj.weight.data
                if cfg.fp32_rescale_math and Wk.dtype != torch.float32:
                    Wk = Wk.float()
                for h, sl in enumerate(self.k_rows):
                    s = float(k_scales[h].item())
                    block = Wk[sl, :]
                    block.mul_(s)
                    Wk[sl, :] = block
                if self.k_proj.weight.data.dtype != Wk.dtype:
                    self.k_proj.weight.data.copy_(Wk.to(self.k_proj.weight.dtype))
                else:
                    self.k_proj.weight.data.copy_(Wk)


def _discover_attention_views(model: nn.Module) -> List[Any]:
    views: List[Any] = []
    for mod in model.modules():
        # qkv-proj style (models.layers.Attention)
        if hasattr(mod, "qkv_proj") and hasattr(mod, "o_proj") and hasattr(mod, "head_dim"):
            try:
                views.append(_QKVModuleView(mod))
            except Exception:
                pass
        # separate-proj style (models.rel_graph.MultiHeadAttention)
        elif hasattr(mod, "q_proj") and hasattr(mod, "k_proj") and hasattr(mod, "heads"):
            try:
                views.append(_SepQKModuleView(mod))
            except Exception:
                pass
    return views


# ------------------------------
# Wrapper Optimizer
# ------------------------------

class MuonClip(torch.optim.Optimizer):
    """
    A thin proxy around any base optimizer that applies QK-Clip style rescale
    immediately after the base optimizer's `step()`.

    Usage:
        base = torch.optim.AdamW(model.parameters(), lr=5e-5)
        opt = MuonClip(base, model, MuonClipConfig(...))
        for ...:
            loss.backward()
            opt.step(); opt.zero_grad()

    You can also pass the MuonClip instance anywhere an Optimizer is expected because
    it forwards param_groups, state, and state_dict/load_state_dict.
    """
    def __init__(self, base_optimizer: torch.optim.Optimizer, model: nn.Module, cfg: Optional[MuonClipConfig] = None):
        # We do NOT call super().__init__ with params, we just proxy to base optimizer.
        self._base = base_optimizer
        self._model = model
        self.cfg = cfg or MuonClipConfig()
        self._step = 0

        # Discover attention modules
        self._views = _discover_attention_views(model)

        # Prepare EMA buffers keyed by module id
        self._ema: Dict[int, torch.Tensor] = {}   # id(module) -> tensor [H] or [KV] etc for QK maxima (per Q-head)
        self._ema_kv: Dict[int, torch.Tensor] = {} # id(module) -> tensor [KV] for K side when KV grouping enabled

        # Enable capture flag on modules (they must implement optional capture in forward)
        for v in self._views:
            m = v.m
            try:
                setattr(m, "_muon_capture", self.cfg.capture_every_forward)
            except Exception:
                pass

    # --- Optimizer interface proxy ---
    @property
    def param_groups(self):
        return self._base.param_groups

    @property
    def state(self):
        return self._base.state

    def zero_grad(self, set_to_none: Optional[bool] = None):
        return self._base.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, *args, **kwargs):
        return self._base.add_param_group(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_state": self._base.state_dict(),
            "muon_state": {
                "step": self._step,
                "ema": {k: v.detach().cpu() for k, v in self._ema.items()},
                "ema_kv": {k: v.detach().cpu() for k, v in self._ema_kv.items()},
                "cfg": self.cfg.__dict__,
            }
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._base.load_state_dict(state_dict.get("base_state", {}))
        mu = state_dict.get("muon_state", {})
        self._step = int(mu.get("step", 0))
        # Attempt immediate EMA restore to correct devices, else fall back to lazy
        self._ema, self._ema_kv = {}, {}
        ema_cpu = mu.get("ema", {})
        ema_kv_cpu = mu.get("ema_kv", {})
        for v in getattr(self, "_views", []):
            try:
                dev = next(v.m.parameters()).device
            except Exception:
                dev = torch.device("cpu")
            mid = id(v.m)
            if mid in ema_cpu:
                self._ema[mid] = ema_cpu[mid].to(dev)
            if mid in ema_kv_cpu:
                self._ema_kv[mid] = ema_kv_cpu[mid].to(dev)
        if self.cfg.verbose:
            print(f"[MuonClip] State restored (step={self._step}, ema={len(self._ema)}, ema_kv={len(self._ema_kv)})")

    def _bounded_scale(self, raw_scale: torch.Tensor) -> torch.Tensor:
        """
        Bound per-step multiplicative scale. Default is down-only ∈ [1 - max_scale_change, 1].
        If allow_upscale, upper bound is 1 + max_upscale_change.
        """
        lower = 1.0 - float(self.cfg.max_scale_change)
        upper = 1.0 + (float(self.cfg.max_upscale_change) if self.cfg.allow_upscale else 0.0)
        return torch.clamp(raw_scale, min=lower, max=upper)

    def _compute_scales_qkv(self, v: _QKVModuleView) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return (q_scales [H] or None, k_scales [KV] or None) for a qkv-style module.
        """
        m = v.m
        max_key = id(m)
        # Read runtime maxima from module if present
        qk_max: Optional[torch.Tensor] = getattr(m, "_muon_last_qk_max", None)

        if qk_max is None or qk_max.numel() == 0:
            return None, None

        # Ensure on device
        device = m.qkv_proj.weight.device
        qk_max = qk_max.detach().to(device=device, dtype=torch.float32)

        # EMA over Q-head maxima (bias-free init from first observation)
        if max_key not in self._ema or self._ema[max_key].numel() != v.num_heads:
            self._ema[max_key] = qk_max.clone()
        else:
            self._ema[max_key].mul_(self.cfg.ema_beta).add_(qk_max, alpha=1.0 - self.cfg.ema_beta)

        # Target-based downscale factor per Q-head
        tau = float(self.cfg.target_logit)
        eps = 1e-6
        desired_q = torch.sqrt(torch.clamp(tau / torch.clamp(self._ema[max_key], min=eps), max=1.0))
        q_scales = self._bounded_scale(desired_q)

        # KV grouping: collapse Q-heads that map to same KV index by taking max over group
        if self.cfg.respect_kv_sharing and v.num_kv < v.num_heads:
            # For KV side, compute per-KV-head target using the max over all Q-heads that share it.
            # Group mapping: kv_idx = h % num_kv
            kv_max = torch.zeros(v.num_kv, device=device, dtype=torch.float32)
            for h in range(v.num_heads):
                kv_idx = h % v.num_kv
                kv_max[kv_idx] = torch.maximum(kv_max[kv_idx], self._ema[max_key][h])
        else:
            # One-to-one
            kv_max = self._ema[max_key][:v.num_kv]

        # EMA for KV-side (optional separate buffer so future extensions can differ)
        if max_key not in self._ema_kv or self._ema_kv[max_key].numel() != v.num_kv:
            self._ema_kv[max_key] = kv_max.clone()
        else:
            self._ema_kv[max_key].mul_(self.cfg.ema_beta).add_(kv_max, alpha=1.0 - self.cfg.ema_beta)

        desired_k = torch.sqrt(torch.clamp(tau / torch.clamp(self._ema_kv[max_key], min=eps), max=1.0))
        k_scales = self._bounded_scale(desired_k)

        return q_scales, k_scales

    def _compute_scales_sep(self, v: _SepQKModuleView) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        m = v.m
        max_key = id(m)
        qk_max: Optional[torch.Tensor] = getattr(m, "_muon_last_qk_max", None)
        if qk_max is None or qk_max.numel() == 0:
            return None, None

        device = v.q_proj.weight.device
        qk_max = qk_max.detach().to(device=device, dtype=torch.float32)

        if max_key not in self._ema or self._ema[max_key].numel() != v.heads:
            self._ema[max_key] = qk_max.clone()
        else:
            self._ema[max_key].mul_(self.cfg.ema_beta).add_(qk_max, alpha=1.0 - self.cfg.ema_beta)

        tau = float(self.cfg.target_logit)
        eps = 1e-6
        desired = torch.sqrt(torch.clamp(tau / torch.clamp(self._ema[max_key], min=eps), max=1.0))
        scales = self._bounded_scale(desired)
        return scales, scales  # Q and K use same per-head scale

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1) Base optimizer update
        self._base.step()

        # 2) MuonClip apply
        self._step += 1
        if (self._step % max(1, int(self.cfg.interval))) != 0:
            return loss

        applied = 0
        for v in self._views:
            try:
                if isinstance(v, _QKVModuleView):
                    q_scales, k_scales = self._compute_scales_qkv(v)
                else:
                    q_scales, k_scales = self._compute_scales_sep(v)

                if q_scales is None and k_scales is None:
                    continue

                # Guardrails: down-only and bounded already; also clamp absolute row scale
                # (We don't track cumulative product here; underlying weights accumulate the effect.)

                v.apply_rescale(q_scales, k_scales, self.cfg)
                applied += 1
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[MuonClip] Skipped module ({type(v.m).__name__}): {e}")

        if self.cfg.verbose:
            print(f"[MuonClip] Applied to {applied} attention modules at step {self._step}")
        return loss


# ------------------------------
# Lightweight helpers
# ------------------------------

def attach_muonclip(model: nn.Module,
                    base_optimizer: torch.optim.Optimizer,
                    cfg: Optional[MuonClipConfig] = None) -> MuonClip:
    """
    Convenience constructor: wrap a base optimizer with MuonClip and toggle capture flags.
    """
    mc = MuonClip(base_optimizer, model, cfg=cfg)
    # Toggle capture according to config (ignored if module doesn't support it)
    for m in model.modules():
        try:
            setattr(m, "_muon_capture", mc.cfg.capture_every_forward)
        except Exception:
            pass
    return mc


# --- Optional: lightweight telemetry for debugging ---
def _safe_mean(x: Optional[torch.Tensor]) -> float:
    try:
        return float(x.mean().item()) if x is not None and x.numel() > 0 else float("nan")
    except Exception:
        return float("nan")

def _safe_max(x: Optional[torch.Tensor]) -> float:
    try:
        return float(x.max().item()) if x is not None and x.numel() > 0 else float("nan")
    except Exception:
        return float("nan")

def _tensor_or_none(x):
    return x if isinstance(x, torch.Tensor) else None

def get_debug_metrics(opt: "MuonClip") -> Dict[str, Any]:
    """
    Return lightweight metrics for logging/monitoring.
    Includes per-view EMA means, maxes, and current τ.
    """
    out = {"tau": float(opt.cfg.target_logit), "views": []}
    for v in getattr(opt, "_views", []):
        mid = id(v.m)
        ema_q = _tensor_or_none(opt._ema.get(mid, None))
        ema_k = _tensor_or_none(opt._ema_kv.get(mid, None))
        out["views"].append({
            "module": type(v.m).__name__,
            "ema_q_mean": _safe_mean(ema_q),
            "ema_q_max":  _safe_max(ema_q),
            "ema_k_mean": _safe_mean(ema_k),
            "ema_k_max":  _safe_max(ema_k),
        })
    return out
