"""
Base class for all pretraining environments.

PretrainBase implements the common EnvironmentProtocol structure:
  - enter/exit session lifecycle
  - step_async/step_wait with token prediction
  - CertSignal published when mastery criteria are met
  - Graduation condition: loss plateau detection

Subclasses implement:
  _get_batch()           → (tokens, targets) for one training step
  _compute_loss(logits, targets, brain) → loss tensor
  cert_name              → certification name issued on graduation
  _mastery_met(loss_ema) → bool, when to issue CertSignal

Each environment controls exactly which parameters are unfrozen:
  TextAlignEnv:       embed table + SSM + language head
  VisionAlignEnv:     SigLIP projection + SSM only (backbone frozen)
  AudioAlignEnv:      Whisper projection + SSM only (backbone frozen)
  VisionFineTuneEnv:  SigLIP last-N layers + projection + SSM
  AudioFineTuneEnv:   Whisper last-N layers + projection + SSM
  DistillEnv:         entire brain (all parameters)
"""
from __future__ import annotations

import math
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch

from ecoframe.protocol import (
    ActionBundle, CapacityError, HardwareSpec,
    SensorBundle, SensorManifest, SensorSpec, Session,
)
from ecoframe.signal import CertSignal, EnvironmentSignal


PRETRAIN_MANIFEST = SensorManifest(
    env_id="pretrain",
    sensors=(
        SensorSpec("text_tokens", (512,),  dtype="int32",
                   action_affected=True, world_external=True),
        SensorSpec("loss",        (1,),    dtype="float32",
                   action_affected=True, world_external=True),
        SensorSpec("phase",       (1,),    dtype="float32",
                   action_affected=False, world_external=False),
    ),
)


class PretrainBase:
    """
    Base pretraining environment. Subclass and implement the abstract methods.
    """

    cert_name:     str   = "pretrained"
    mastery_loss:  float = 2.0     # CE loss threshold to issue cert
    patience:      int   = 2000    # steps of no improvement before cert check
    hardware_spec: HardwareSpec = HardwareSpec.cpu()
    manifest:      SensorManifest = PRETRAIN_MANIFEST

    def __init__(
        self,
        brain,                     # UniversalBrain or CortexModel
        optimizer,                 # torch optimizer
        device:     torch.device,
        field       = None,        # ecoframe Field for cert publishing
        verbose:    bool = False,
    ):
        self.brain     = brain
        self.optimizer = optimizer
        self.device    = device
        self.field     = field
        self.verbose   = verbose

        self._sessions:    dict[str, Session] = {}
        self._step_count   = 0
        self._loss_ema     = 6.0
        self._loss_prev    = 6.0
        self._plateau_steps = 0
        self._cert_issued  = False
        self._pending_tokens: torch.Tensor | None = None
        self.env_id = f"pretrain_{self.cert_name}"
        self.capacity = 1

    # ── EnvironmentProtocol ────────────────────────────────────────────────────

    def start(self) -> None:
        self._unfreeze_parameters()
        if self.verbose:
            n_train = sum(p.numel() for p in self.brain.parameters()
                         if p.requires_grad)
            print(f"{self.env_id}: {n_train:,} trainable params", flush=True)

    def close(self) -> None:
        pass

    def enter(self, brain_id: str, ssm_state: dict | None = None) -> Session:
        if len(self._sessions) >= self.capacity:
            raise CapacityError(f"{self.env_id}: capacity reached")
        session = Session(brain_id=brain_id, env_id=self.env_id,
                          agent_id=brain_id, ssm_state=ssm_state or {},
                          entered_at=self._step_count)
        self._sessions[brain_id] = session
        return session

    def exit(self, session: Session) -> dict:
        self._sessions.pop(session.brain_id, None)
        return session.ssm_state

    def reset(self, session: Session) -> dict[str, SensorBundle]:
        return {session.brain_id: self._make_bundle(session.brain_id, 0.0)}

    def step_async(self, actions: dict[str, ActionBundle]) -> None:
        """Receive brain's predicted tokens (or just trigger next step)."""
        self._pending_actions = actions

    def step_wait(self) -> dict[str, SensorBundle]:
        """Execute one training step, return loss as reward."""
        self._step_count += 1
        bundles = {}

        for brain_id, session in self._sessions.items():
            tokens, targets = self._get_batch()
            loss = self._train_step(tokens, targets)
            loss_val = float(loss)

            # EMA tracking
            self._loss_ema = 0.97 * self._loss_ema + 0.03 * loss_val

            # Plateau detection
            if self._loss_prev - self._loss_ema > 0.01:
                self._plateau_steps = 0
            else:
                self._plateau_steps += 1
            self._loss_prev = self._loss_ema

            # Mastery check
            done = False
            if (not self._cert_issued and
                    self._plateau_steps >= self.patience and
                    self._mastery_met(self._loss_ema)):
                self._issue_cert(brain_id, session)
                done = True

            if self._step_count % 100 == 0:
                self._publish_env_signal()
                if self.verbose:
                    print(f"{self.env_id} step={self._step_count:,} "
                          f"loss_ema={self._loss_ema:.4f}", flush=True)

            bundles[brain_id] = self._make_bundle(brain_id, loss_val, done)

        self._pending_actions = {}
        return bundles

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_tokens, target_tokens) tensors on device."""
        ...

    @abstractmethod
    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute scalar loss from model logits and targets."""
        ...

    def _unfreeze_parameters(self) -> None:
        """Override to control which parameters are trainable."""
        pass   # default: don't change any requires_grad

    def _mastery_met(self, loss_ema: float) -> bool:
        return loss_ema < self.mastery_loss

    # ── Training step ─────────────────────────────────────────────────────────

    def _train_step(
        self, tokens: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        self.optimizer.zero_grad()
        raw = self._get_raw()
        pos = torch.zeros(*tokens.shape, 3, dtype=torch.long, device=self.device)
        with torch.enable_grad():
            logits = raw(tokens, pos)
            loss   = self._compute_loss(logits, targets)
        if loss.isfinite() and float(loss.detach().abs()) > 1e-10:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g['params']], 1.0)
            self.optimizer.step()
        return loss.detach()

    def _get_raw(self):
        b = self.brain
        if hasattr(b, 'cortex'):  b = b.cortex   # UniversalBrain
        if hasattr(b, '_orig_mod'): b = b._orig_mod
        elif hasattr(b, 'module'):  b = b.module
        return b

    # ── Certification ──────────────────────────────────────────────────────────

    def _issue_cert(self, brain_id: str, session: Session) -> None:
        self._cert_issued = True
        if self.field is None:
            return
        cert_env_id = f"cert_{self.cert_name}"
        self.field.register_agent(cert_env_id, pos=(4.0, float(hash(self.cert_name) % 10)))
        sig = CertSignal(
            position          = (4.0, 0.0),
            timestamp         = self._step_count,
            publisher         = cert_env_id,
            brain_id          = brain_id,
            cert_name         = self.cert_name,
            passed            = 1.0,
            score             = max(0.0, 1.0 - self._loss_ema / 6.0),
            retry_after_steps = 5000,
        )
        self.field.publish(cert_env_id, sig)
        if self.verbose:
            print(f"{self.env_id}: CERT '{self.cert_name}' issued "
                  f"(loss={self._loss_ema:.4f})", flush=True)

    def _publish_env_signal(self) -> None:
        if self.field is None:
            return
        self.field.register_agent(self.env_id, pos=(0.0, 0.0))
        self.field.publish(self.env_id, EnvironmentSignal(
            position      = (0.0, 0.0),
            timestamp     = self._step_count,
            publisher     = self.env_id,
            curiosity     = self._loss_ema,
            load_fraction = len(self._sessions) / max(1, self.capacity),
            env_type      = self.env_id,
        ))

    def _make_bundle(
        self, brain_id: str, loss_val: float, done: bool = False
    ) -> SensorBundle:
        return SensorBundle(
            extra          = {'text_tokens': np.zeros(512, dtype=np.int32),
                              'loss':        np.array([loss_val], dtype=np.float32)},
            proprioceptive = np.array([self._step_count / 100_000.0], dtype=np.float32),
            reward         = -loss_val,   # lower loss = higher reward
            done           = done,
            env_id         = self.env_id,
            agent_id       = brain_id,
            step           = self._step_count,
        )
