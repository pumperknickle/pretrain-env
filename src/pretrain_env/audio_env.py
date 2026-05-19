"""
Audio pretraining environments.

AudioAlignEnv:     Whisper projection + SSM only. Backbone frozen.
                   Cert: "audio_aligned"

AudioFineTuneEnv:  Unfreezes last N Whisper layers + projection + SSM.
                   Cert: "audio_finetuned"
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn.functional as F

from pretrain_env.base import PretrainBase


class AudioAlignEnv(PretrainBase):
    """
    Align Whisper → projection → SSM on speech-text data.
    Backbone frozen. Only projection + SSM train.
    """

    cert_name    = "audio_aligned"
    patience     = 500

    def __init__(self, brain, optimizer, device, batch_size=2,
                 field=None, verbose=False):
        super().__init__(brain, optimizer, device, field, verbose)
        self.env_id   = "pretrain_audio_align"
        self.batch_sz = batch_size
        self._data_iter = None

    def start(self) -> None:
        if hasattr(self.brain, 'whisper') and self.brain.whisper is not None:
            self.brain.whisper.load(self.device)
        self._unfreeze_parameters()
        self._data_iter = self._build_data_iter()
        if self.verbose:
            print("AudioAlignEnv: Whisper backbone frozen, training projection + SSM",
                  flush=True)

    def _unfreeze_parameters(self) -> None:
        raw = self._get_raw()
        for p in raw.parameters():
            p.requires_grad_(True)
        # Unfreeze Whisper projection only
        if hasattr(self.brain, 'whisper') and self.brain.whisper:
            self.brain.whisper.proj.requires_grad_(True)
            self.brain.whisper.norm.requires_grad_(True)
            if self.brain.whisper._encoder:
                for p in self.brain.whisper._encoder.parameters():
                    p.requires_grad_(False)

    def _train_step(self, mel, transcript_ids):
        """Encode audio via Whisper, predict transcript tokens."""
        self.optimizer.zero_grad()
        raw = self._get_raw()

        with torch.enable_grad():
            if hasattr(self.brain, 'whisper') and self.brain.whisper:
                a_tokens, a_pos = self.brain.encode_audio(mel)
            else:
                d = raw.cfg.d_model
                a_tokens = torch.zeros(mel.shape[0], 100, d, device=self.device,
                                       dtype=next(raw.parameters()).dtype)
                a_pos    = torch.zeros(mel.shape[0], 100, 3, dtype=torch.long,
                                       device=self.device)

            t_tokens, t_pos = self.brain.encode_text(transcript_ids[:, :-1])
            all_tokens = torch.cat([a_tokens, t_tokens], dim=1)
            all_pos    = torch.cat([a_pos,    t_pos],    dim=1)

            empty = torch.zeros(mel.shape[0], 0, dtype=torch.long, device=self.device)
            _, _, _, h_all = raw.forward_stateful(
                idx=empty, extra_embeds=[(all_tokens, all_pos)])
            h_text = h_all[:, -transcript_ids.shape[1]+1:, :]
            logits = raw.head(h_text)

            V = logits.shape[-1]
            loss = F.cross_entropy(
                logits.reshape(-1, V),
                transcript_ids[:, 1:].reshape(-1),
                ignore_index=-1,
            )

        if loss.isfinite():
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g['params']], 1.0)
            self.optimizer.step()

        return loss.detach()

    def _get_batch(self): return next(self._data_iter)
    def _compute_loss(self, logits, targets):
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), ignore_index=-1)

    def _build_data_iter(self):
        from brain.tokenizer import get_tokenizer
        tok = get_tokenizer()
        dev = self.device
        bsz = self.batch_sz

        phrases = [
            "the car turned left at the intersection",
            "execute the following python function",
            "the robot navigates through the corridor",
            "return the sum of all elements in the list",
            "the traffic light changed from red to green",
        ]

        def _gen():
            while True:
                mel = torch.randn(bsz, 80, 3000, device=dev)  # fake mel
                txt = random.choice(phrases)
                ids = tok.encode(txt)[:64] + [0] * (64 - len(tok.encode(txt)[:64]))
                t   = torch.tensor([ids] * bsz, dtype=torch.long, device=dev)
                yield mel, t

        return _gen()


class AudioFineTuneEnv(AudioAlignEnv):
    """Fine-tune last N Whisper encoder layers. Requires: 'audio_aligned'."""

    cert_name    = "audio_finetuned"
    patience     = 500
    last_n_layers = 2

    def __init__(self, *args, last_n_layers: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_id       = "pretrain_audio_finetune"
        self.last_n_layers = last_n_layers

    def _unfreeze_parameters(self) -> None:
        super()._unfreeze_parameters()
        whisper = getattr(self.brain, 'whisper', None)
        if whisper and whisper._encoder:
            enc = whisper._encoder
            if hasattr(enc, 'layers'):
                for layer in list(enc.layers)[-self.last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad_(True)
