# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deprecated module path retained for backward compatibility.

The model wrapper used to live here as ``DiffusionLightningModule``. It has
moved to :mod:`mattergen.diffusion.model_module` and been renamed to
``DiffusionModelModule``. Importing from this module emits a
``DeprecationWarning`` and re-exports the canonical names. The old class name
is preserved as an alias so that pickled checkpoints and YAML configs whose
``_target_`` references ``mattergen.diffusion.lightning_module.DiffusionLightningModule``
continue to load.
"""

from __future__ import annotations

import warnings

from mattergen.diffusion.model_module import (
    DiffusionLightningModule,
    DiffusionModelModule,
    OptimizerPartial,
    SchedulerPartial,
)

warnings.warn(
    "mattergen.diffusion.lightning_module is deprecated; "
    "import from mattergen.diffusion.model_module instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DiffusionLightningModule",
    "DiffusionModelModule",
    "OptimizerPartial",
    "SchedulerPartial",
]
