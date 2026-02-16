# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging
import os
import random
import re
from glob import glob
from pathlib import Path
from typing import Any, Mapping, TypeVar

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from mattergen.common.utils.config_utils import get_config
from mattergen.diffusion.config import Config, resolve_model_module_cfg
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.exceptions import AmbiguousConfig
from mattergen.diffusion.native_ddp import fit as native_fit

T = TypeVar("T")

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def maybe_instantiate(instance_or_config: T | Mapping, expected_type=None, **kwargs) -> T:
    """
    If instance_or_config is a mapping with a _target_ field, instantiate it.
    Otherwise, return it as is.
    """
    if isinstance(instance_or_config, Mapping) and "_target_" in instance_or_config:
        instance = instantiate(instance_or_config, **kwargs)
    else:
        instance = instance_or_config
    assert expected_type is None or isinstance(
        instance, expected_type
    ), f"Expected {expected_type}, got {type(instance)}"
    return instance


def _find_latest_checkpoint(dirpath: str) -> str | None:
    """Finds the most recent checkpoint inside `dirpath`."""

    # checkpoint names are like "epoch=0-step=0.ckpt."
    # Find the checkpoint with highest epoch:
    def extract_epoch(ckpt):
        match = re.search(r"epoch=(\d+)", ckpt)
        if match:
            return int(match.group(1))
        return -1

    ckpts = glob(f"{dirpath}/*.ckpt")
    epochs = np.array([extract_epoch(ckpt) for ckpt in ckpts])
    if len(epochs) == 0 or epochs.max() < 0:
        # No checkpoints found.
        return None
    latest_checkpoint = ckpts[epochs.argmax()]
    return latest_checkpoint


def main(
    config: Config | DictConfig, save_config: bool = True, seed: int | None = None
) -> tuple[None, Any]:
    """
    Main entry point to train and evaluate a diffusion model.

    save_config: if True, the config will be saved both as a YAML file and in each checkpoint. This doesn't work if the config contains things that can't be `yaml.dump`-ed, so
    if you don't care about saving and loading checkpoints and want to use a config that contains things like `torch.nn.Module`s already instantiated, set this to False.
    """
    if config.checkpoint_path and config.auto_resume:
        raise AmbiguousConfig(
            f"Ambiguous config: you set both a checkpoint path {config.checkpoint_path} and `auto_resume` which means automatically select a checkpoint path to resume from."
        )

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    trainer_backend = config.get("trainer_backend", "native_ddp")
    if trainer_backend != "native_ddp":
        raise ValueError(
            "Only native PyTorch DDP training is supported. "
            "Set trainer_backend=native_ddp."
        )

    if isinstance(config, DictConfig):
        config_as_dict = OmegaConf.to_container(config, resolve=True)
    else:
        raise NotImplementedError

    datamodule = maybe_instantiate(config.data_module)

    ckpt_path = config.checkpoint_path
    if config.auto_resume:
        ckpt_path = _find_latest_checkpoint(str(Path(os.getcwd()) / "checkpoints"))

    model_module_cfg = resolve_model_module_cfg(config)
    diffusion_module: DiffusionModule = instantiate(model_module_cfg.diffusion_module)
    optimizer_partial = maybe_instantiate(model_module_cfg.get("optimizer_partial"))
    scheduler_partials_cfg = model_module_cfg.get("scheduler_partials", [])
    scheduler_partials = [
        {
            **scheduler_dict,
            "scheduler": maybe_instantiate(scheduler_dict["scheduler"]),
        }
        for scheduler_dict in scheduler_partials_cfg
    ]

    native_fit(
        diffusion_module=diffusion_module,
        datamodule=datamodule,
        trainer_cfg=config.trainer,
        native_cfg=config.native_trainer,
        config_dict=config_as_dict,
        ckpt_path=ckpt_path,
        optimizer_partial=optimizer_partial,
        scheduler_partials=scheduler_partials,
    )

    return None, diffusion_module


def cli(argv: list[str] | None) -> None:
    """
    Args:
        argv: list of command-line arguments as strings, or None. If None,
          command-line arguments will be got from sys.argv
    """

    parser = argparse.ArgumentParser(allow_abbrev=False)  # prevent prefix matching issues
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to use. If not provided, a random seed will be used.",
    )
    args, argv = parser.parse_known_args(argv)

    # Create config from command-line arguments.
    config = get_config(argv, Config)
    main(config, seed=args.seed)


if __name__ == "__main__":
    cli(argv=None)
