# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import Any, Mapping

from hydra.core.hydra_config import HydraConfig


@dataclass
class Config:

    # This is for CLI applications that need to reuse a CLI parameter in multiple places
    # in the config file. The idea is that you use `my_cli params.output_dir=foobar`
    # and in other places in the config file `output_dir: ${params.output_dir}`
    params: dict[str, Any] = field(default_factory=dict)

    checkpoint_path: str | None = None  # Required if train == False

    # if load_original is True then we load original weights in validation mode instead of EMA
    load_original: bool = False

    # When auto_resume is set to `True` the trainer saves a copy of each checkpoint in
    # {trainer.default_root_dir}/checkpoints. Before starting training, we look in this
    # directory for a checkpoint from which to resume training.
    auto_resume: bool = False

    # Training backend.
    trainer_backend: str = "native_ddp"

    # Canonical model module config namespace.
    model_module: dict[str, Any] = field(default_factory=dict)

    # Backward-compatible alias for older configs.
    lightning_module: dict[str, Any] = field(default_factory=dict)

    # Trainer settings consumed by native DDP training.
    trainer: dict[str, Any] = field(default_factory=dict)

    # Native PyTorch DDP trainer settings.
    native_trainer: dict[str, Any] = field(default_factory=dict)

    # LightningDataModule
    data_module: dict[str, Any] = field(default_factory=dict)


def _cfg_get(cfg: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key)
    return getattr(cfg, key, None)


def resolve_model_module_cfg(cfg: Mapping[str, Any] | Any) -> Any:
    model_module_cfg = _cfg_get(cfg, "model_module")
    legacy_cfg = _cfg_get(cfg, "lightning_module")

    has_model_module_cfg = model_module_cfg is not None and bool(model_module_cfg)
    has_legacy_cfg = legacy_cfg is not None and bool(legacy_cfg)

    if not has_model_module_cfg and not has_legacy_cfg:
        raise ValueError(
            "Missing model module config. Expected `model_module` (or legacy `lightning_module`)."
        )
    if not has_model_module_cfg:
        return legacy_cfg
    if not has_legacy_cfg:
        return model_module_cfg

    prefer_legacy = False
    prefer_model = False
    try:
        overrides = HydraConfig.get().overrides.task
    except Exception:
        overrides = []

    for override in overrides:
        normalized = override.lstrip("+~")
        if normalized.startswith("lightning_module."):
            prefer_legacy = True
        elif normalized.startswith("model_module."):
            prefer_model = True

    if prefer_legacy and not prefer_model:
        return legacy_cfg
    return model_module_cfg
