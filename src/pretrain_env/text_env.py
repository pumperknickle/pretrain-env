"""
TextAlignEnv: next-token prediction on text/code corpus.

Trains: SSM weights + language head + embedding table.
Frozen:  SigLIP, Whisper (they're not needed for text).

Cert: "text_aligned" when CE loss < mastery_loss for patience steps.

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
    mastery_loss = 2.5    # achievable in ~10K steps for small model
    patience     = 1000

    def __init__(
        self,
        brain,
        optimizer,
        device:      torch.device,
        data_source: str   = 'code',
        seq_len:     int   = 512,
        batch_size:  int   = 4,
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
        try:
            import datasets as hf
            if self._data_source == 'code':
                ds = hf.load_dataset("codeparrot/github-code", "Python",
                                      split="train", streaming=True,
                                      trust_remote_code=True)
                text_key = 'code'
            else:
                ds = hf.load_dataset("Skylion007/openwebtext",
                                      split="train", streaming=True)
                text_key = 'text'
        except Exception:
            ds = None; text_key = None

        from brain.tokenizer import get_tokenizer
        tok = get_tokenizer()
        seq = self.seq_len
        bsz = self.batch_sz
        dev = self.device

        def _gen():
            buf = []
            if ds is not None:
                it = iter(ds)
                while True:
                    try:
                        row  = next(it)
                        text = row.get(text_key, row.get('text', ''))
                        ids  = tok.encode(text)[:seq * 4]
                        buf.extend(ids)
                        while len(buf) >= (seq + 1) * bsz:
                            chunk = buf[:(seq+1)*bsz]
                            buf   = buf[seq:]
                            t = torch.tensor(chunk, dtype=torch.long, device=dev)
                            yield t.view(bsz, seq + 1)
                    except StopIteration:
                        it = iter(ds)
            else:
                V = 33024
                while True:
                    ids = [random.randint(0, V-1) for _ in range((seq+1)*bsz)]
                    yield torch.tensor(ids, dtype=torch.long, device=dev).view(bsz, seq+1)

        return _gen()
