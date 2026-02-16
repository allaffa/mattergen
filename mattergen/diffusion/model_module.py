# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mattergen.diffusion.lightning_module import DiffusionLightningModule

# Canonical name for the model wrapper module.
DiffusionModelModule = DiffusionLightningModule

__all__ = ["DiffusionModelModule", "DiffusionLightningModule"]
