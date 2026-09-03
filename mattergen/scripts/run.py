# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import faulthandler
import logging

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf

from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.common.utils.rank_debug import trace_rank
from mattergen.diffusion.config import Config
from mattergen.diffusion.run import main

logger = logging.getLogger(__name__)


@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "conf"), config_name="default", version_base="1.1"
)
def mattergen_main(cfg: omegaconf.DictConfig):
    faulthandler.enable(all_threads=True)
    trace_rank("mattergen_main_entered")
    # Tensor Core acceleration (leads to ~2x speed-up during training)
    torch.set_float32_matmul_precision("high")
    # Make merged config options
    # CLI options take priority over YAML file options
    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, cfg)
    OmegaConf.set_readonly(config, True)  # should not be written to
    trace_rank("hydra_config_resolved")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    try:
        main(config, seed=config.seed)
    except BaseException as exc:
        trace_rank("training_exception", error=repr(exc), error_type=type(exc).__name__)
        raise


if __name__ == "__main__":
    mattergen_main()
