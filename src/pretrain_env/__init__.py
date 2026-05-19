"""
pretrain-env: pretraining and distillation environments as ecoframe EnvironmentProtocol.

Full pretraining pipeline for UniversalBrain:

  Stage 1: Alignment (encoders frozen)
    TextAlignEnv    → cert "text_aligned"
    VisionAlignEnv  → cert "vision_aligned"
    AudioAlignEnv   → cert "audio_aligned"

  Stage 2: Encoder fine-tuning (last N layers unfrozen)
    VisionFineTuneEnv → cert "vision_finetuned"  [requires: vision_aligned]
    AudioFineTuneEnv  → cert "audio_finetuned"   [requires: audio_aligned]

  Stage 3: Distillation (full brain)
    DistillEnv        → cert "distilled_{teacher}"

  Stage 4: Growth (optional — scale up while preserving world model)
    GrowthEnv(nano→small) → cert "graduated_to_small"

All environments work through MetaEnvironment + TrainingEngine.
Each issues CertSignal on mastery — BrainRegistry gates next environment.

Quick start (small brain on RTX 3090):
    from pretrain_env import build_pretrain_pipeline
    from brain.universal_brain import UniversalBrain

    brain = UniversalBrain.build(scale='nano')
    pipeline = build_pretrain_pipeline(brain, target_scale='small',
                                       teacher='Qwen/Qwen2.5-7B-Instruct')
    for env in pipeline:
        # register in MetaEnvironment and train until cert issued
        meta.register_env(env)
"""
from pretrain_env.text_env    import TextAlignEnv
from pretrain_env.vision_env  import VisionAlignEnv, VisionFineTuneEnv
from pretrain_env.audio_env   import AudioAlignEnv, AudioFineTuneEnv
from pretrain_env.distill_env import DistillEnv
from pretrain_env.growth_env  import GrowthEnv, migrate_ssm_state

__version__ = "0.1.0"
__all__ = [
    "TextAlignEnv",
    "VisionAlignEnv", "VisionFineTuneEnv",
    "AudioAlignEnv",  "AudioFineTuneEnv",
    "DistillEnv",
    "GrowthEnv", "migrate_ssm_state",
    "build_pretrain_pipeline",
]


def build_pretrain_pipeline(
    brain,
    optimizer,
    device,
    target_scale:  str  = 'small',
    teacher_id:    str  = 'Qwen/Qwen2.5-7B-Instruct',
    data_source:   str  = 'code',
    field          = None,
    verbose:       bool = True,
) -> list:
    """
    Build the ordered list of pretraining environments for a brain.

    Returns environments in training order. Register each in MetaEnvironment
    via meta.register_env(env). BrainRegistry handles cert-gated progression.

    For scale='small' on RTX 3090, the full pipeline takes ~8-12 hours.
    """
    envs = [
        TextAlignEnv(brain, optimizer, device, data_source=data_source,
                     field=field, verbose=verbose),
        VisionAlignEnv(brain, optimizer, device, field=field, verbose=verbose),
        AudioAlignEnv(brain, optimizer, device, field=field, verbose=verbose),
        VisionFineTuneEnv(brain, optimizer, device, field=field, verbose=verbose),
        AudioFineTuneEnv(brain, optimizer, device, field=field, verbose=verbose),
        DistillEnv(brain, optimizer, device, teacher_id=teacher_id,
                   data_source=data_source, field=field, verbose=verbose),
    ]

    # Add growth stage if we need to scale up
    from brain.model import _SCALE_CONFIGS, make_brain
    raw = brain
    if hasattr(raw, 'cortex'):  raw = raw.cortex
    if hasattr(raw, '_orig_mod'): raw = raw._orig_mod
    elif hasattr(raw, 'module'): raw = raw.module

    current_d = raw.cfg.d_model
    target_d  = _SCALE_CONFIGS[target_scale][0]
    if target_d > current_d:
        envs.append(GrowthEnv(brain, optimizer, device, scale_to=target_scale,
                              data_source=data_source, field=field, verbose=verbose))
    return envs
