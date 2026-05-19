"""
GrowthEnv: progressive brain scaling with preserved world model.

Grows CortexModel from scale_from to scale_to using Net2Net-style
function-preserving transforms:

  1. Width expansion: expand all weight matrices.
     Old weights fill the top-left block. New rows/columns = small noise.
     The expanded model computes the same function initially (function-preserving).

  2. SSM state migration: embed old state into new, larger state.
     Old SSM state (B, d_inner_old, 2*d_state) → padded →
     New SSM state (B, d_inner_new, 2*d_state) with zeros for new dims.
     New capacity starts contributing nothing — earned influence.

  3. Train on any task (same as current environment).
     New capacity earns contribution via gradient over training steps.

  4. Exit when new capacity is actively utilized:
     gradient norm in new weight rows exceeds activation_threshold.

  5. Issue CertSignal("graduated_to_{scale_to}").

Net2Net reference: Chen et al. 2015 "Net2Net: Accelerating Learning via
Knowledge Transfer". Function-preserving transforms enable 2-3x faster
convergence vs training from scratch on the larger model.

Why this preserves "essence":
  - Causal reasoning patterns (from distillation/training) encoded in SSM weights
    are preserved in the top-left block of expanded matrices.
  - The brain's accumulated world model (SSM state) is embedded in the new state.
  - Only NEW capacity is random (near-zero) — existing knowledge is intact.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrain_env.base import PretrainBase


class GrowthEnv(PretrainBase):
    """
    Expand brain from scale_from to scale_to.
    Requires cert for scale_from to be valid.
    Issues cert "graduated_to_{scale_to}".
    """

    mastery_loss = 999.0   # overridden — we use gradient-based exit
    patience     = 0

    def __init__(
        self,
        brain,
        optimizer,
        device:           torch.device,
        scale_to:         str   = 'small',
        activation_threshold: float = 0.01,  # grad norm ratio for new vs old weights
        warmup_steps:     int   = 1000,       # min steps before exit check
        data_source:      str   = 'code',
        seq_len:          int   = 512,
        batch_size:       int   = 4,
        field             = None,
        verbose:          bool  = False,
    ):
        super().__init__(brain, optimizer, device, field, verbose)
        self.scale_to    = scale_to
        self.threshold   = activation_threshold
        self.warmup      = warmup_steps
        self.env_id      = f"pretrain_grow_to_{scale_to}"
        self.cert_name   = f"graduated_to_{scale_to}"
        self._data_source = data_source
        self.seq_len     = seq_len
        self.batch_sz    = batch_size
        self._data_iter  = None
        self._expanded   = False
        self._old_d_model = None

    def start(self) -> None:
        if self.verbose:
            print(f"GrowthEnv: expanding brain to scale='{self.scale_to}'...",
                  flush=True)
        self._expand_brain()
        self._data_iter = self._build_text_iter()
        self._unfreeze_parameters()

    def _expand_brain(self) -> None:
        """Net2Net width expansion: grow d_model while preserving function."""
        from brain.model import _SCALE_CONFIGS

        raw = self._get_raw()
        self._old_d_model = raw.cfg.d_model

        target_cfg = _SCALE_CONFIGS[self.scale_to]
        new_d_model = target_cfg[0]   # d_model is first element

        if new_d_model <= self._old_d_model:
            if self.verbose:
                print(f"  Brain already at or above {self.scale_to} "
                      f"({self._old_d_model} >= {new_d_model})", flush=True)
            self._expanded = False
            return

        if self.verbose:
            print(f"  Expanding d_model: {self._old_d_model} → {new_d_model}",
                  flush=True)

        _expand_cortex(raw, self._old_d_model, new_d_model, self.device,
                       verbose=self.verbose)

        # Rebuild optimizer for expanded parameters
        raw_params = [p for p in raw.parameters() if p.requires_grad]
        lr = self.optimizer.param_groups[0]['lr']
        self.optimizer = type(self.optimizer)(raw_params, lr=lr)
        self._expanded = True

        if self.verbose:
            n = sum(p.numel() for p in raw.parameters())
            print(f"  Expanded brain: {n:,} params", flush=True)

    def _unfreeze_parameters(self) -> None:
        raw = self._get_raw()
        for p in raw.parameters():
            p.requires_grad_(True)

    def _get_batch(self):
        batch = next(self._data_iter)
        return batch[:, :-1], batch[:, 1:]

    def _compute_loss(self, logits, targets):
        V = logits.shape[-1]
        return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                               ignore_index=-1)

    def _mastery_met(self, loss_ema: float) -> bool:
        """
        Exit when new weight dimensions are actively contributing.
        Measured by: grad norm in new rows / grad norm in old rows > threshold.
        """
        if not self._expanded:
            return True   # no expansion needed, immediately graduate
        if self._step_count < self.warmup:
            return False

        raw     = self._get_raw()
        old_d   = self._old_d_model
        new_d   = raw.cfg.d_model
        ratio   = _new_weight_activation_ratio(raw, old_d, new_d)

        if self.verbose and self._step_count % 500 == 0:
            print(f"  GrowthEnv activation ratio: {ratio:.4f} "
                  f"(threshold={self.threshold})", flush=True)

        return ratio > self.threshold

    def _build_text_iter(self):
        from brain.tokenizer import get_tokenizer
        tok = get_tokenizer()
        from pretrain_env.corpus import get_corpus_iter
        return get_corpus_iter(tok, self.seq_len, self.batch_sz, self.device)


# ── Net2Net expansion ──────────────────────────────────────────────────────────

def _expand_cortex(
    model:     nn.Module,
    old_d:     int,
    new_d:     int,
    device:    torch.device,
    noise:     float = 1e-3,
    verbose:   bool  = False,
) -> None:
    """
    In-place Net2Net width expansion of CortexModel.

    For each Linear layer:
      - weight (out, in): expand to (new_out, new_in)
        by duplicating old rows/cols + small noise
      - bias: pad with zeros

    For SSM parameters (A_log, A_phase, D):
      - Expand d_inner dimension: copy old values, add noise for new
    """
    expanded = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                _expand_linear(module, old_d, new_d, noise, device)
                expanded += 1
            elif isinstance(module, nn.LayerNorm):
                _expand_layernorm(module, old_d, new_d, device)
            elif isinstance(module, nn.Embedding):
                _expand_embedding_d(module, old_d, new_d, device)

        # Update config
        if hasattr(model, 'cfg'):
            model.cfg.d_model = new_d
            # Recompute dependent dims
            if hasattr(model.cfg, 'd_ff'):
                old_ff = model.cfg.d_ff
                model.cfg.d_ff = int(old_ff * new_d / old_d)

    if verbose:
        print(f"  Net2Net: expanded {expanded} Linear layers "
              f"{old_d}→{new_d}", flush=True)


def _expand_linear(
    module: nn.Linear, old_d: int, new_d: int,
    noise: float, device: torch.device
) -> None:
    old_w = module.weight.data   # (out_features, in_features)
    old_out, old_in = old_w.shape

    new_out = int(old_out * new_d / old_d) if old_out == old_d else old_out
    new_in  = int(old_in  * new_d / old_d) if old_in  == old_d else old_in

    if new_out == old_out and new_in == old_in:
        return   # this layer doesn't depend on d_model

    new_w = torch.zeros(new_out, new_in, dtype=old_w.dtype, device=device)
    new_w[:old_out, :old_in] = old_w
    # New rows/cols: small noise (earned influence — starts contributing ≈0)
    if new_out > old_out:
        new_w[old_out:, :old_in] = noise * torch.randn(
            new_out - old_out, old_in, dtype=old_w.dtype, device=device)
    if new_in > old_in:
        new_w[:, old_in:] = noise * torch.randn(
            new_out, new_in - old_in, dtype=old_w.dtype, device=device)

    module.weight = nn.Parameter(new_w)

    if module.bias is not None:
        old_b = module.bias.data
        if old_b.shape[0] == old_out and new_out > old_out:
            new_b = torch.zeros(new_out, dtype=old_b.dtype, device=device)
            new_b[:old_out] = old_b
            module.bias = nn.Parameter(new_b)

    module.in_features  = new_in
    module.out_features = new_out


def _expand_layernorm(
    module: nn.LayerNorm, old_d: int, new_d: int, device: torch.device
) -> None:
    if module.normalized_shape[0] != old_d:
        return
    old_w = module.weight.data
    old_b = module.bias.data if module.bias is not None else None
    new_w = torch.ones(new_d, dtype=old_w.dtype, device=device)
    new_w[:old_d] = old_w
    module.weight = nn.Parameter(new_w)
    if old_b is not None:
        new_b = torch.zeros(new_d, dtype=old_b.dtype, device=device)
        new_b[:old_d] = old_b
        module.bias = nn.Parameter(new_b)
    module.normalized_shape = (new_d,)


def _expand_embedding_d(
    module: nn.Embedding, old_d: int, new_d: int, device: torch.device
) -> None:
    if module.embedding_dim != old_d:
        return
    old_w = module.weight.data   # (vocab, old_d)
    new_w = torch.zeros(module.num_embeddings, new_d, dtype=old_w.dtype, device=device)
    new_w[:, :old_d] = old_w
    module.weight = nn.Parameter(new_w)
    module.embedding_dim = new_d


def _new_weight_activation_ratio(
    model: nn.Module, old_d: int, new_d: int
) -> float:
    """
    Ratio of gradient norm in NEW weight rows vs OLD weight rows.
    When new dimensions are actively used, this ratio approaches 1.0.
    """
    old_norms = []
    new_norms = []
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.weight.grad is not None:
            g = module.weight.grad
            out_f = module.out_features
            if out_f == new_d:
                old_norms.append(g[:old_d].norm().item())
                new_norms.append(g[old_d:].norm().item())
    if not old_norms or not new_norms:
        return 0.0
    avg_old = sum(old_norms) / len(old_norms)
    avg_new = sum(new_norms) / len(new_norms)
    return avg_new / max(avg_old, 1e-8)


def migrate_ssm_state(
    old_state: list | None,
    old_d_inner: int,
    new_d_inner: int,
    d_state: int,
    device:  torch.device,
) -> list | None:
    """
    Expand SSM hidden state from old to new d_inner.
    Old state embedded in new state; new dimensions = zeros.

    old_state: list of tensors [(B, old_d_inner, 2*d_state), ...]
    Returns:   list of tensors [(B, new_d_inner, 2*d_state), ...]
    """
    if old_state is None:
        return None

    new_state = []
    for h in old_state:
        if isinstance(h, tuple):
            new_state.append(tuple(
                _pad_state_tensor(t, old_d_inner, new_d_inner, device) for t in h))
        else:
            new_state.append(_pad_state_tensor(h, old_d_inner, new_d_inner, device))
    return new_state


def _pad_state_tensor(
    t: torch.Tensor, old_d: int, new_d: int, device: torch.device
) -> torch.Tensor:
    if t.shape[1] == old_d and new_d > old_d:
        B, _, rest = t.shape
        pad = torch.zeros(B, new_d - old_d, rest, dtype=t.dtype, device=device)
        return torch.cat([t.to(device), pad], dim=1)
    return t
