# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Storage backends for cached crystal datasets.

MatterGen historically persisted its cached datasets as a directory of
:mod:`numpy` ``.npy`` files plus per-property ``.json`` files. The classes in
this package replace that layout with two HydraGNN-inspired backends:

* :mod:`mattergen.common.data.backends.pickle_backend` — a simple
  ``dataset.pkl`` per split. Always available.
* :mod:`mattergen.common.data.backends.adios_backend` — an ADIOS2 ``.bp``
  file. Requires the optional ``adios2`` Python package and is loaded lazily.

Both backends operate on a :class:`CacheBundle` of plain :class:`numpy`
arrays. The cached payload is what the on-disk file holds; in-memory PyG
``ChemGraph`` (a subclass of :class:`torch_geometric.data.Data`) objects are
materialised lazily by :class:`mattergen.common.data.dataset.CrystalDataset`.
"""

from mattergen.common.data.backends._bundle import (
    CacheBundle,
    PickleFormat,
    AdiosFormat,
    SUPPORTED_FORMATS,
    concat_bundles,
    detect_format,
    list_available_properties,
    read_cache,
    write_cache,
)
from mattergen.common.data.backends._mpi import (
    SingleRankComm,
    auto_comm,
    is_mpi_comm,
    nsplit,
    shard_range,
)

__all__ = [
    "CacheBundle",
    "PickleFormat",
    "AdiosFormat",
    "SUPPORTED_FORMATS",
    "SingleRankComm",
    "auto_comm",
    "concat_bundles",
    "detect_format",
    "is_mpi_comm",
    "list_available_properties",
    "nsplit",
    "read_cache",
    "shard_range",
    "write_cache",
]
