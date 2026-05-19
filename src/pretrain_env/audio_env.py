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

    def _train_step(self, audio_features_or_mel, transcript_ids):
        """
        audio_features_or_mel: either pre-computed Whisper encoder output
          (B, T_audio, D_whisper) OR raw mel (B, 80, T_frames).
        If pre-computed (3D with last dim != 80), skip Whisper encoding —
        this gives 10-50x speedup since Whisper is the bottleneck.
        """
        self.optimizer.zero_grad()
        raw = self._get_raw()
        model_dtype = next(raw.parameters()).dtype
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

        with torch.enable_grad(), torch.autocast(device_type=device_type, dtype=model_dtype):
            inp = audio_features_or_mel

            # Detect pre-encoded features vs raw mel (80 mel bins = raw)
            is_preencoded = (inp.ndim == 3 and inp.shape[1] != 80)

            if is_preencoded:
                # Already encoded — just apply projection
                B = inp.shape[0]
                if hasattr(self.brain, 'whisper') and self.brain.whisper:
                    a_raw = inp.to(self.device).float()
                    a_tokens = self.brain.whisper.proj(
                        self.brain.whisper.norm(a_raw)).to(model_dtype)
                else:
                    a_tokens = inp.to(model_dtype)
                a_pos = torch.zeros(B, a_tokens.shape[1], 3,
                                    dtype=torch.long, device=self.device)
            elif hasattr(self.brain, 'whisper') and self.brain.whisper:
                a_tokens, a_pos = self.brain.encode_audio(inp)
                a_tokens = a_tokens.to(model_dtype)
            else:
                d = raw.cfg.d_model
                B = inp.shape[0]
                a_tokens = torch.zeros(B, 100, d, device=self.device, dtype=model_dtype)
                a_pos    = torch.zeros(B, 100, 3, dtype=torch.long, device=self.device)

            t_tokens, t_pos = self.brain.encode_text(transcript_ids[:, :-1])
            t_tokens = t_tokens.to(model_dtype)
            all_tokens = torch.cat([a_tokens, t_tokens], dim=1)
            all_pos    = torch.cat([a_pos,    t_pos],    dim=1)

            empty = torch.zeros(inp.shape[0], 0, dtype=torch.long, device=self.device)
            _, _, _, h_all = raw.forward_stateful(
                idx=empty, extra_embeds=[(all_tokens, all_pos)])
            n_cap = transcript_ids.shape[1] - 1   # input tokens = T-1
            h_text = h_all[:, -n_cap:, :]
            logits = raw.head(h_text.to(model_dtype))

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
        """
        Real speech synthesis → real Whisper encoder features.
        Philosophy 2: real data, not synthetic noise.

        Uses espeak-ng TTS to synthesize the Python corpus into actual audio,
        then runs through the real Whisper encoder once and caches the features.
        The brain learns genuine audio-text correspondence.
        """
        from brain.tokenizer import get_tokenizer
        from pretrain_env.corpus import PYTHON_CORPUS
        tok = get_tokenizer()
        dev = self.device
        bsz = self.batch_sz

        # Generate real audio features from actual text
        audio_features, phrase_tokens = _generate_real_audio_features(
            texts=PYTHON_CORPUS[:20],   # first 20 corpus snippets
            tokenizer=tok,
            whisper_adapter=getattr(self.brain, 'whisper', None),
            device=dev,
            verbose=self.verbose,
        )

        if self.verbose:
            print(f"  AudioAlignEnv: {len(audio_features)} real speech samples ready",
                  flush=True)

        def _gen():
            while True:
                idx = random.randrange(len(audio_features))
                feat = audio_features[idx].to(dev).expand(bsz, -1, -1)
                t    = torch.tensor([phrase_tokens[idx]] * bsz,
                                     dtype=torch.long, device=dev)
                yield feat, t

        return _gen()


def _generate_real_audio_features(
    texts: list[str],
    tokenizer,
    whisper_adapter,
    device: 'torch.device',
    verbose: bool = False,
) -> tuple[list, list]:
    """
    Synthesize real speech from text using espeak-ng, encode with Whisper.

    Returns:
        audio_features: list of (1, T, D_whisper) tensors — real encoder outputs
        phrase_tokens:  list of int lists — tokenized transcripts
    """
    import subprocess, tempfile, os
    import torch

    # Check espeak-ng available
    has_espeak = subprocess.run(['which', 'espeak-ng'], capture_output=True).returncode == 0
    if not has_espeak:
        # Try installing
        subprocess.run(['apt-get', 'install', '-y', '-q', 'espeak-ng'], capture_output=True)
        has_espeak = subprocess.run(['which', 'espeak-ng'], capture_output=True).returncode == 0

    if not has_espeak:
        raise RuntimeError(
            "espeak-ng not available. Install with: apt-get install -y espeak-ng\n"
            "AudioAlignEnv requires real speech synthesis. "
            "Cannot use synthetic noise — see CLAUDE.md Philosophy 2.")

    if verbose:
        print(f"  Synthesizing {len(texts)} speech samples via espeak-ng...", flush=True)

    audio_features = []
    phrase_tokens  = []

    for i, text in enumerate(texts):
        # Clean text — espeak works best with simple descriptions, not raw code
        # For code snippets, extract the docstring or first comment
        clean = _extract_speakable(text)
        if not clean:
            continue

        # Synthesize to WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
        try:
            result = subprocess.run(
                ['espeak-ng', '-w', wav_path, '-s', '150', clean],
                capture_output=True, timeout=10)
            if result.returncode != 0 or not os.path.exists(wav_path):
                continue

            # Load audio and create mel spectrogram
            feat = _wav_to_whisper_features(wav_path, whisper_adapter, device)
            if feat is None:
                continue

            audio_features.append(feat)
            ids = tokenizer.encode(clean[:128])[:64]
            ids += [0] * (64 - len(ids))
            phrase_tokens.append(ids)

            if verbose and i % 5 == 0:
                print(f"    [{i+1}/{len(texts)}] encoded: {clean[:40]!r}", flush=True)
        except Exception as e:
            if verbose:
                print(f"    [{i+1}/{len(texts)}] failed: {e}", flush=True)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    if not audio_features:
        raise RuntimeError(
            "AudioAlignEnv: failed to generate any real audio features. "
            "Check espeak-ng installation and Whisper encoder.")

    return audio_features, phrase_tokens


def _extract_speakable(code: str) -> str:
    """Extract a speakable description from a code snippet."""
    lines = code.strip().split('\n')
    for line in lines:
        line = line.strip()
        # Prefer function/class definition lines
        if line.startswith('def ') or line.startswith('class '):
            name = line.split('(')[0].replace('def ', '').replace('class ', '')
            # Convert snake_case to words
            words = name.replace('_', ' ')
            return f"define function {words}"
        # Or docstrings/comments
        if line.startswith('"""') or line.startswith("'''"):
            text = line.strip('"\' ')
            if len(text) > 10:
                return text[:100]
        if line.startswith('#'):
            return line[1:].strip()[:100]
    # Fallback: first non-empty line
    for line in lines:
        if line.strip() and not line.startswith('import'):
            return line.strip()[:80]
    return ""


def _wav_to_whisper_features(
    wav_path: str,
    whisper_adapter,
    device: 'torch.device',
) -> 'torch.Tensor | None':
    """Load WAV, get real Whisper encoder features."""
    import torch
    try:
        import librosa
        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    except ImportError:
        # Fallback: use soundfile
        try:
            import soundfile as sf
            audio, sr = sf.read(wav_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import resampy
                audio = resampy.resample(audio, sr, 16000)
        except Exception:
            return None

    import numpy as np
    if len(audio) < 400:   # too short
        return None

    # Use Whisper's feature extractor
    if whisper_adapter is not None and getattr(whisper_adapter, '_loaded', False):
        from transformers import WhisperFeatureExtractor
        try:
            extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
            features  = extractor(audio.astype(np.float32), sampling_rate=16000,
                                   return_tensors="pt")
            mel = features.input_features.to(device)
            with torch.no_grad():
                out = whisper_adapter._encoder(input_features=mel)
            return out.last_hidden_state.detach().cpu()
        except Exception:
            pass

    # Fallback if Whisper unavailable: at least use real audio statistics
    # (still wrong but better than pure noise)
    return None


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
