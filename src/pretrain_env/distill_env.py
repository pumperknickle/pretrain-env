"""
DistillEnv: knowledge distillation from teacher LLM → CortexModel SSM.

Full brain unfrozen. Teacher runs frozen at 4-bit quantization.
Loss: α×CE(hard targets) + β×KL(teacher soft targets, T=4)

For small brain on RTX 3090:
  Teacher Qwen2.5-7B (4-bit ≈ 4GB) + Student small (≈ 1GB) = fits easily.

Cert: "distilled_{teacher_name}" e.g. "distilled_qwen2.5_7b"
"""
from __future__ import annotations

import re
import torch
import torch.nn.functional as F

from pretrain_env.base import PretrainBase


class DistillEnv(PretrainBase):
    """
    Teacher LLM → student CortexModel distillation.
    Full brain unfrozen. Soft + hard targets.
    """

    mastery_loss = 2.0
    patience     = 3000

    def __init__(
        self,
        brain,
        optimizer,
        device:      torch.device,
        teacher_id:  str   = 'Qwen/Qwen2.5-7B-Instruct',
        data_source: str   = 'code',
        seq_len:     int   = 512,
        batch_size:  int   = 4,
        alpha:       float = 0.4,
        beta:        float = 0.6,
        temperature: float = 4.0,
        field        = None,
        verbose:     bool  = False,
    ):
        super().__init__(brain, optimizer, device, field, verbose)
        self.teacher_id  = teacher_id
        self._data_source = data_source
        self.seq_len     = seq_len
        self.batch_sz    = batch_size
        self.alpha       = alpha
        self.beta        = beta
        self.temperature = temperature
        self._teacher    = None
        self._data_iter  = None

        # cert name based on teacher
        slug = re.sub(r'[^a-z0-9]', '_', teacher_id.lower().split('/')[-1])
        self.cert_name = f"distilled_{slug}"
        self.env_id    = f"pretrain_distill_{slug}"

    def start(self) -> None:
        if self.verbose:
            print(f"DistillEnv: loading teacher {self.teacher_id}...", flush=True)
        self._teacher   = self._load_teacher()
        self._data_iter = self._build_data_iter()
        self._unfreeze_parameters()
        if self.verbose:
            print(f"DistillEnv: ready  α={self.alpha} CE + β={self.beta} KL "
                  f"T={self.temperature}", flush=True)

    def _unfreeze_parameters(self) -> None:
        """Unfreeze everything — distillation trains the full brain."""
        raw = self._get_raw()
        for p in raw.parameters():
            p.requires_grad_(True)
        # Also unfreeze projection layers on encoders
        for name in ['siglip', 'whisper']:
            adapter = getattr(self.brain, name, None)
            if adapter:
                adapter.proj.requires_grad_(True)
                adapter.norm.requires_grad_(True)

    def _train_step(self, tokens, targets):
        """Override: also compute teacher logits for KL loss."""
        self.optimizer.zero_grad()
        raw = self._get_raw()
        V   = raw.cfg.vocab_size

        with torch.no_grad():
            t_out    = self._teacher['model'](input_ids=tokens)
            t_logits = t_out.logits.float()[..., :V]

        pos = torch.zeros(*tokens.shape, 3, dtype=torch.long, device=self.device)
        with torch.enable_grad():
            s_logits = raw(tokens, pos)

            ce_l = F.cross_entropy(s_logits.reshape(-1, V), targets.reshape(-1),
                                   ignore_index=-1)
            T    = self.temperature
            kl_l = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction='batchmean') * (T ** 2)
            loss = self.alpha * ce_l + self.beta * kl_l

        if loss.isfinite():
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g['params']], 1.0)
            self.optimizer.step()

        return loss.detach()

    def _get_batch(self):
        batch = next(self._data_iter)
        return batch[:, :-1], batch[:, 1:]

    def _compute_loss(self, logits, targets):
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), ignore_index=-1)

    def _load_teacher(self) -> dict:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
        tok   = AutoTokenizer.from_pretrained(self.teacher_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.teacher_id, quantization_config=bnb,
            device_map='auto', trust_remote_code=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return {'model': model, 'tokenizer': tok}

    def _build_data_iter(self):
        import random
        try:
            import datasets as hf
            ds = hf.load_dataset("codeparrot/github-code", "Python",
                                  split="train", streaming=True, trust_remote_code=True)
            text_key = 'code'
        except Exception:
            ds = None; text_key = None

        from brain.tokenizer import get_tokenizer
        tok = get_tokenizer()
        seq = self.seq_len; bsz = self.batch_sz; dev = self.device

        def _gen():
            buf = []
            if ds is not None:
                it = iter(ds)
                while True:
                    try:
                        row  = next(it)
                        ids  = tok.encode(row.get(text_key,''))[:seq*4]
                        buf.extend(ids)
                        while len(buf) >= (seq+1)*bsz:
                            chunk = buf[:(seq+1)*bsz]; buf = buf[seq:]
                            yield torch.tensor(chunk, dtype=torch.long, device=dev).view(bsz, seq+1)
                    except StopIteration:
                        it = iter(ds)
            else:
                while True:
                    ids = [random.randint(0, 33023) for _ in range((seq+1)*bsz)]
                    yield torch.tensor(ids, dtype=torch.long, device=dev).view(bsz, seq+1)

        return _gen()
