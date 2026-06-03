# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import Any, Mapping


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

    # Model module config namespace.
    model_module: dict[str, Any] = field(default_factory=dict)

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
    """Return the canonical ``model_module`` config block.

    Raises ``ValueError`` if it is missing or empty.
    """

    # loading in from legacy config files can cause some issues
    # this doesnt cause problems in training bc in training 
    # it is assumed the user correctly prescribed all the configs
    # and then model is simply loaded from the legacy checkpoints
    # for generation the 'model path' needs to be prescribed which
    # is used to read BOTH the config (containing legacy fields)
    # and the checkpoints. the below works reasonably well despite being
    # a little hacky

    if 'lightning_module' in list(cfg.keys()):
        model_module_cfg = _cfg_get(cfg, "lightning_module")
    else:
        model_module_cfg = _cfg_get(cfg, "model_module")
    if not model_module_cfg:
        raise ValueError(
            "Missing model module config: expected a non-empty `model_module` block."
        )
    return model_module_cfg
