# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any

from mattergen.common.data.property_scalers import compute_property_scalers
from mattergen.denoiser import GemNetTDenoiser
from mattergen.diffusion.model_module import DiffusionModelModule


class SetPropertyScalers:
    """
    Utility callback; at the start of training, this computes the mean and std of the property data and adds the property
    scalers to the model.
    """

    def on_fit_start(self, trainer: Any, pl_module: DiffusionModelModule):
        model: GemNetTDenoiser = pl_module.diffusion_module.model

        # model.property_embeddings: torch.nn.ModuleDict always exists
        compute_property_scalers(datamodule=trainer.datamodule, property_embeddings=model.property_embeddings)

        if hasattr(model, "property_embeddings_adapt"):
            # this is a fine tune model
            compute_property_scalers(
                datamodule=trainer.datamodule, property_embeddings=model.property_embeddings_adapt
            )
