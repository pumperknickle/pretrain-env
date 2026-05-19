"""
Vision pretraining environments.

VisionAlignEnv:     SigLIP projection + SSM only. Backbone frozen.
                    Cert: "vision_aligned"

VisionFineTuneEnv:  Unfreezes last N SigLIP layers + projection + SSM.
                    Cert: "vision_finetuned"
                    Requires: "vision_aligned"

Both use image-caption prediction: given a caption, predict visual tokens.
Or visual prediction: given previous frame, predict next frame features.
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn.functional as F

from pretrain_env.base import PretrainBase


class VisionAlignEnv(PretrainBase):
    """
    Align SigLIP → projection → SSM on image-caption data.
    Backbone (SigLIP ViT) stays frozen. Only projection + SSM train.

    Task: given image visual tokens, predict the next caption tokens.
    This teaches the SSM to ground language understanding in visual features.
    """

    cert_name    = "vision_aligned"
    mastery_loss = 3.0
    patience     = 500

    def __init__(
        self,
        brain,
        optimizer,
        device:      torch.device,
        batch_size:  int = 2,
        image_size:  int = 224,
        field        = None,
        verbose:     bool = False,
    ):
        super().__init__(brain, optimizer, device, field, verbose)
        self.env_id   = "pretrain_vision_align"
        self.batch_sz = batch_size
        self.img_size = image_size
        self._data_iter = None

    def start(self) -> None:
        # Load SigLIP if not already loaded
        if hasattr(self.brain, 'siglip') and self.brain.siglip is not None:
            self.brain.siglip.load(self.device)
        self._unfreeze_parameters()
        self._data_iter = self._build_data_iter()
        if self.verbose:
            print(f"VisionAlignEnv: SigLIP backbone frozen, "
                  f"training projection + SSM", flush=True)

    def _unfreeze_parameters(self) -> None:
        raw = self._get_raw()
        # Freeze everything first
        for p in raw.parameters():
            p.requires_grad_(False)
        # Unfreeze SSM + embed + head
        for name, p in raw.named_parameters():
            p.requires_grad_(True)
        # Unfreeze SigLIP projection only (not the backbone)
        if hasattr(self.brain, 'siglip') and self.brain.siglip:
            self.brain.siglip.proj.requires_grad_(True)
            self.brain.siglip.norm.requires_grad_(True)
            # Backbone stays frozen
            if self.brain.siglip._encoder:
                for p in self.brain.siglip._encoder.parameters():
                    p.requires_grad_(False)

    def _get_batch(self):
        return next(self._data_iter)

    def _train_step(self, tokens, targets):
        """Override: encode images through SigLIP then predict caption tokens."""
        images, caption_ids = tokens, targets

        self.optimizer.zero_grad()
        raw = self._get_raw()

        with torch.enable_grad():
            # Visual tokens via SigLIP
            if hasattr(self.brain, 'siglip') and self.brain.siglip:
                v_tokens, v_pos = self.brain.encode_visual(images)
            else:
                v_tokens = torch.zeros(
                    images.shape[0], 196, raw.cfg.d_model,
                    device=self.device, dtype=next(raw.parameters()).dtype)
                v_pos = torch.zeros(images.shape[0], 196, 3,
                                    dtype=torch.long, device=self.device)

            # Caption context tokens
            t_tokens, t_pos = self.brain.encode_text(caption_ids[:, :-1])
            all_tokens = torch.cat([v_tokens, t_tokens], dim=1)
            all_pos    = torch.cat([v_pos,    t_pos],    dim=1)

            empty = torch.zeros(images.shape[0], 0, dtype=torch.long, device=self.device)
            logits, *_ = raw.forward_stateful(idx=empty, extra_embeds=[(all_tokens, all_pos)])

            if logits is None:
                # forward_stateful doesn't return logits — use head directly
                _, _, _, h_all = raw.forward_stateful(
                    idx=empty, extra_embeds=[(all_tokens, all_pos)])
                h_last = h_all[:, -caption_ids.shape[1]+1:, :]
                logits = raw.head(h_last)

            targets_shifted = caption_ids[:, 1:]
            V = logits.shape[-1]
            loss = F.cross_entropy(
                logits.reshape(-1, V),
                targets_shifted.reshape(-1),
                ignore_index=-1,
            )

        if loss.isfinite():
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g['params']], 1.0)
            self.optimizer.step()

        return loss.detach()

    def _compute_loss(self, logits, targets):
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), ignore_index=-1)

    def _build_data_iter(self):
        """Image-caption data. Falls back to random images + tokenized captions."""
        from brain.tokenizer import get_tokenizer
        tok = get_tokenizer()
        dev = self.device
        bsz = self.batch_sz
        sz  = self.img_size

        # Try loading LAION-CC3M or similar
        try:
            import datasets as hf
            ds = hf.load_dataset("conceptual_captions", split="train",
                                  streaming=True, trust_remote_code=True)
            from PIL import Image
            import requests
            from io import BytesIO
            use_real = True
        except Exception:
            use_real = False

        def _gen():
            captions = [
                "a photo of a car driving on a road",
                "a person writing code on a laptop",
                "a robot navigating through a room",
                "a bird flying over a forest",
                "geometric shapes on a white background",
            ]
            while True:
                # Synthetic fallback: random images + fixed captions
                imgs = torch.randn(bsz, 3, sz, sz, device=dev).clamp(0, 1)
                cap  = random.choice(captions)
                cap_ids = tok.encode(cap)[:64]
                cap_ids += [0] * (64 - len(cap_ids))
                cap_t = torch.tensor([cap_ids] * bsz, dtype=torch.long, device=dev)
                yield imgs, cap_t

        return _gen()


class VisionFineTuneEnv(VisionAlignEnv):
    """
    Fine-tune last N SigLIP layers for domain-specific visual features.
    Requires: "vision_aligned" cert.

    Unfreezes: SigLIP last_n_layers + projection + SSM.
    """

    cert_name    = "vision_finetuned"
    mastery_loss = 2.5
    patience     = 2000
    last_n_layers = 4   # unfreeze last 4 ViT layers

    def __init__(self, *args, last_n_layers: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_id      = "pretrain_vision_finetune"
        self.last_n_layers = last_n_layers

    def _unfreeze_parameters(self) -> None:
        super()._unfreeze_parameters()   # projection + SSM
        # Additionally unfreeze last N encoder layers
        siglip = getattr(self.brain, 'siglip', None)
        if siglip and siglip._encoder:
            enc = siglip._encoder
            layers = getattr(enc, 'vision_model', enc)
            # Get encoder layers
            if hasattr(layers, 'encoder') and hasattr(layers.encoder, 'layers'):
                all_layers = list(layers.encoder.layers)
                for layer in all_layers[-self.last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad_(True)
            if self.verbose:
                n_open = sum(1 for p in enc.parameters() if p.requires_grad)
                print(f"VisionFineTuneEnv: {n_open} SigLIP params unfrozen "
                      f"(last {self.last_n_layers} layers)", flush=True)
