from __future__ import annotations

from collections import defaultdict
from typing import Any, TypeVar

import torch
from tqdm.auto import tqdm

TensorOrStringType = TypeVar("TensorOrStringType", torch.Tensor, list[str])


def maybe_to_tensor(values: list[TensorOrStringType]) -> TensorOrStringType:
    if isinstance(values[0], torch.Tensor):
        return torch.cat(values)
    # chemical system is str and therefore cannot be converted to tensor
    return [el for x in values for el in x]


def compute_property_scalers(datamodule: Any, property_embeddings: torch.nn.ModuleDict):
    property_values = defaultdict(list)

    # property names may be distinct from keys in this dictionary
    property_names = [p.name for p in property_embeddings.values() if not isinstance(p.scaler, torch.nn.Identity)]
    if len(property_names) == 0:
        return

    train_loader = datamodule.train_dataloader()
    for batch in tqdm(train_loader, desc="Fitting property scalers"):
        for property_name in property_names:
            # concat all values in train dataset for this given property
            property_values[property_name].append(batch[property_name])

    for property_name in property_names:
        property_embeddings[property_name].fit_scaler(
            all_data=maybe_to_tensor(values=property_values[property_name])
        )
