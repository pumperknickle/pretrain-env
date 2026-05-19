"""
TextAlignEnv: next-token prediction on text/code corpus.

Trains: SSM weights + language head + embedding table.
Frozen:  SigLIP, Whisper (they're not needed for text).


Data sources (streaming, no full download):
  'code':  codeparrot/github-code (Python subset)
  'text':  Skylion007/openwebtext
  'mixed': both interleaved
"""
from __future__ import annotations

import random
import torch
import torch.nn.functional as F

from pretrain_env.base import PretrainBase


class TextAlignEnv(PretrainBase):
    """Next-token prediction. Trains SSM + embed + head. SigLIP/Whisper frozen."""

    cert_name    = "text_aligned"
    patience     = 500

    def __init__(
        self,
        brain,
        optimizer,
        device:      torch.device,
        data_source: str   = 'code',
        seq_len:     int   = 128,   # shorter = faster iterations on nano model
        batch_size:  int   = 8,     # larger batch compensates for shorter seq
        field        = None,
        verbose:     bool  = False,
    ):
        super().__init__(brain, optimizer, device, field, verbose)
        self.env_id   = "pretrain_text_align"
        self.seq_len  = seq_len
        self.batch_sz = batch_size
        self._data_iter = None
        self._data_source = data_source

    def start(self) -> None:
        self._unfreeze_parameters()
        self._data_iter = self._build_data_iter()
        if self.verbose:
            print(f"TextAlignEnv: data={self._data_source} seq={self.seq_len}",
                  flush=True)

    def _unfreeze_parameters(self) -> None:
        raw = self._get_raw()
        # Unfreeze SSM + embed + head. Keep SigLIP/Whisper frozen.
        for name, p in raw.named_parameters():
            p.requires_grad_(True)
        # Freeze encoder backbones if present on the UniversalBrain
        for adapter_name in ['siglip', 'whisper']:
            adapter = getattr(self.brain, adapter_name, None)
            if adapter and hasattr(adapter, '_encoder') and adapter._encoder:
                for p in adapter._encoder.parameters():
                    p.requires_grad_(False)

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch = next(self._data_iter)   # (B, L+1)
        return batch[:, :-1], batch[:, 1:]

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        V = logits.shape[-1]
        return F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=-1,
        )

    def _build_data_iter(self):
        """
        Fast corpus-only iterator. Pre-tokenized once at startup.
        No network, no threads, no locks — pure CPU tensor ops.
        """
        from brain.tokenizer import get_tokenizer
        from pretrain_env.corpus import get_corpus_iter
        tok = get_tokenizer()
        if self.verbose:
            from pretrain_env.corpus import PYTHON_CORPUS
            print(f"  TextAlignEnv: corpus ({len(PYTHON_CORPUS)} snippets, "
                  f"seq={self.seq_len} batch={self.batch_sz})", flush=True)
        return get_corpus_iter(tok, self.seq_len, self.batch_sz, self.device)
