import torch
from torch.utils.data import Dataset
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.utils.rank_debug import trace_rank
import adios2
import numpy as np
import re
from threading import Lock


def _parse_adios_shape(value, variable):
    values = re.findall(r"\d+", value) if isinstance(value, str) else value
    shape = tuple(int(item) for item in values)
    if not shape:
        raise ValueError(f"ADIOS variable {variable!r} has no global shape")
    return shape

class HydraGNNAdiosCrystalDataset(Dataset):
    """Read HydraGNN-format ADIOS data without importing all of HydraGNN.

    HydraGNN's top-level package eagerly imports its training, plotting, and
    TensorFlow-related dependencies.  Importing that package concurrently on
    hundreds of Frontier ranks creates a severe shared-filesystem startup
    storm before the ADIOS reader is even constructed.  This class preserves
    the on-disk format and the existing MatterGen-facing API while using an
    independent persistent ADIOS reader on each rank.
    """

    def __init__(
        self,
        filename,
        label,
        comm=None,
        transforms=None,
        properties=None,
        keys=None,
        max_samples=None,
        **_,
    ):
        # ``comm`` remains accepted for configuration compatibility.  Reads
        # are deliberately independent, so dataset construction has no MPI
        # collective that can wait on a rank still importing Python modules.
        del comm
        self.filename = str(filename)
        self.label = str(label)
        self.transforms = transforms
        self.property_names = set(properties or [])
        self.keys = tuple(keys or ("pos", "cell", "atomic_numbers", "forces"))
        self._reader_lock = Lock()
        trace_rank("adios_dataset_init_entered", label=self.label)
        trace_rank("before_adios_file_open", label=self.label)
        self.f = adios2.FileReader(self.filename)
        trace_rank("after_adios_file_open", label=self.label)
        self.variables = self.f.available_variables()

        self.node_count_variable = f"{self.label}/natoms"
        if self.node_count_variable not in self.variables:
            raise KeyError(f"ADIOS variable not found: {self.node_count_variable}")
        source_n_samples = _parse_adios_shape(
            self.variables[self.node_count_variable]["Shape"],
            self.node_count_variable,
        )[0]
        if max_samples is None:
            self.n_samples = source_n_samples
        else:
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError("max_samples must be positive when provided")
            if max_samples > source_n_samples:
                raise ValueError(
                    f"Requested max_samples={max_samples}, but ADIOS split "
                    f"{self.label!r} contains only {source_n_samples} samples"
                )
            self.n_samples = max_samples

        atomic_numbers_variable = f"{self.label}/atomic_numbers"
        if atomic_numbers_variable not in self.variables:
            raise KeyError(f"ADIOS variable not found: {atomic_numbers_variable}")
        atomic_shape = _parse_adios_shape(
            self.variables[atomic_numbers_variable]["Shape"],
            atomic_numbers_variable,
        )
        non_unit_extents = [extent for extent in atomic_shape if extent != 1]
        if len(non_unit_extents) != 1:
            raise ValueError(
                f"Cannot infer total atoms from {atomic_numbers_variable!r} "
                f"with shape {atomic_shape}"
            )
        if self.n_samples == source_n_samples:
            self.total_node_count = int(non_unit_extents[0])
        else:
            count_variable = f"{atomic_numbers_variable}/variable_count"
            offset_variable = f"{atomic_numbers_variable}/variable_offset"
            missing_metadata = [
                name
                for name in (count_variable, offset_variable)
                if name not in self.variables
            ]
            if missing_metadata:
                raise KeyError(
                    "Cannot calculate the atom count for a limited ADIOS prefix; "
                    f"missing variables: {missing_metadata}"
                )
            first_offset = int(
                np.asarray(self.f.read(offset_variable, [0], [1])).reshape(-1)[0]
            )
            last_index = self.n_samples - 1
            last_offset = int(
                np.asarray(self.f.read(offset_variable, [last_index], [1])).reshape(-1)[0]
            )
            last_count = int(
                np.asarray(self.f.read(count_variable, [last_index], [1])).reshape(-1)[0]
            )
            self.total_node_count = last_offset + last_count - first_offset
            if self.total_node_count <= 0:
                raise ValueError(
                    f"Invalid packed atomic-number metadata for {self.label!r} prefix"
                )

        required_keys = {"pos", "cell", "atomic_numbers"}
        if "force_rms" in self.property_names:
            required_keys.add("forces")
        missing = [
            key
            for key in sorted(required_keys)
            if f"{self.label}/{key}" not in self.variables
        ]
        if missing:
            raise KeyError(
                f"ADIOS split {self.label!r} is missing required variables: {missing}"
            )
        trace_rank(
            "adios_dataset_init_completed",
            label=self.label,
            samples=self.n_samples,
            source_samples=source_n_samples,
            total_nodes=self.total_node_count,
        )

    def __len__(self):
        return self.n_samples

    def len(self):
        """Compatibility with HydraGNN's historical dataset interface."""

        return len(self)

    def _read_ragged_sample(self, name, idx):
        variable = f"{self.label}/{name}"
        count_variable = f"{variable}/variable_count"
        offset_variable = f"{variable}/variable_offset"
        count = int(
            np.asarray(self.f.read(count_variable, [idx], [1])).reshape(-1)[0]
        )
        offset = int(
            np.asarray(self.f.read(offset_variable, [idx], [1])).reshape(-1)[0]
        )
        shape = _parse_adios_shape(self.variables[variable]["Shape"], variable)
        start = [0] * len(shape)
        selection = list(shape)
        start[0] = offset
        selection[0] = count
        return np.asarray(self.f.read(variable, start, selection))

    def read_node_counts_range(self, start, count):
        """Read contiguous ``natoms`` metadata without materializing samples."""

        start = int(start)
        count = int(count)
        if start < 0 or count < 0 or start + count > len(self):
            raise IndexError("node-count range is outside the dataset")
        if count == 0:
            return np.empty(0, dtype=np.int64)
        with self._reader_lock:
            return np.asarray(
                self.f.read(self.node_count_variable, [start], [count]),
                dtype=np.int64,
            ).reshape(-1)

    def __getitem__(self, idx) -> ChemGraph:
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        with self._reader_lock:
            atomic_numbers = torch.as_tensor(
                self._read_ragged_sample("atomic_numbers", idx)
            ).reshape(-1).long()
            pos = torch.as_tensor(
                self._read_ragged_sample("pos", idx), dtype=torch.float32
            )
            cell = torch.as_tensor(
                self._read_ragged_sample("cell", idx), dtype=torch.float32
            ).reshape(3, 3)
            forces = None
            if "force_rms" in self.property_names:
                forces = torch.as_tensor(
                    self._read_ragged_sample("forces", idx), dtype=torch.float32
                ).reshape(-1, 3)

        natoms = int(atomic_numbers.numel())
        props = {}
        if forces is not None:
            if forces.shape[0] != natoms:
                raise ValueError(
                    f"Sample {idx} has {natoms} atoms but {forces.shape[0]} force rows"
                )
            force_rms = torch.sqrt(torch.sum(forces.square(), dim=-1).mean())
            props["force_rms"] = force_rms.clamp_min(1e-8).reshape(1)

        data = ChemGraph(
            pos=pos % 1.0,
            cell=cell.unsqueeze(0),
            atomic_numbers=atomic_numbers,
            num_atoms=natoms,
            num_nodes=natoms,
            **props,
        )
        if self.transforms is not None:
            for transform in self.transforms:
                data = transform(data)
        return data

    def get(self, idx):
        """Compatibility with HydraGNN's historical dataset interface."""

        return self[idx]

    def close(self):
        reader = getattr(self, "f", None)
        if reader is not None:
            reader.close()
            self.f = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

class LazyAdiosCrystalDataset(Dataset):
    """
    Map-style dataset for DataLoader/DistributedSampler.

    DistributedSampler chooses integer indices in [0, len(dataset)).
    DataLoader then calls __getitem__(idx), and that is where we lazy-read
    just the one structure requested.

    Note: this is a simple implementation to demonstrate the idea. It opens
    and closes the ADIOS file on every sample, which is probably not the most
    efficient way to do it. A more efficient version would keep one reader open
    per worker process, but that is more complex to implement.
    """
    def __init__(self, path, split, properties=None, transforms=None):
        # Path is location of file
        self.path = path
        # Split is "trainset" or "valset" or "testset"
        self.split = split
        # Save properties and transforms
        self.property_names = properties or []
        self.transforms = transforms
        # Open once here only to read metadata and find the size
        with adios2.FileReader(self.path) as reader:
            self.n_samples = int(reader.available_variables()[f"{self.split}/natoms"]["Shape"])

    def __len__(self):
        # DistributedSampler uses this to know valid indices are 0..n_samples-1.
        return self.n_samples
    
    def _read_scalar_sample(self, reader, name, idx):
        """Read one value from a dense per-sample array, like natoms."""
        value = reader.read(name, start=[idx], count=[1], step_selection=[0, 1])
        return int(np.asarray(value)[0])
    
    def _read_vector_sample(self, reader, name, idx, width):
        """Read N values from a fixed-size per-sample array, like cell/lattice."""
        value = reader.read(name, start=[idx], count=[1, width], step_selection=[0, 1])
        return int(np.asarray(value)[0])

    def _read_ragged_sample(self, reader, name, idx, width):
        """
        Read one sample from a packed ragged array.

        Example: pos for all structures is saved as one big [total_atoms, 3]
        array. The companion arrays say where sample idx starts and how many
        rows it owns:

            name/variable_offset[idx] = first row in the big array
            name/variable_count[idx]  = number of rows for this sample
        """
        count = reader.read(f"{name}/variable_count", start=[idx], count=[1], step_selection=[0, 1])[0]
        offset = reader.read(f"{name}/variable_offset", start=[idx], count=[1], step_selection=[0, 1])[0]

        return np.asarray(reader.read(name, start=[int(offset), 0], count=[int(count), width], step_selection=[0, 1]))

    def _read_properties(self, reader, idx):
        """
        Read all requested properties for this sample.
        Note: We assume all properties are scalar
        """

        props_dict = {}
        if self.property_names is not None:
            for prop in self.property_names:
                count = reader.read(f"{self.split}/{prop}/variable_count", start=[idx], count=[1], step_selection=[0, 1])[0]
                offset = reader.read(f"{self.split}/{prop}/variable_offset", start=[idx], count=[1], step_selection=[0, 1])[0]
                props_dict[prop] = np.asarray(reader.read(f"{self.split}/{prop}", start=[int(offset), 0], count=[int(count), 1], step_selection=[0, 1]))
        return props_dict

    def __getitem__(self, idx) -> ChemGraph:
        # ADIOS wants plain ints.
        idx = int(idx)

        # Open reader per sample (probably not the most efficient)
        with adios2.FileReader(self.path) as reader:
            # Get number of atoms in the structure
            natoms = self._read_scalar_sample(reader, f"{self.split}/natoms", idx)

            # Get the atomic numbers of the atoms in the structure 
            atomic_numbers = self._read_ragged_sample(reader, f"{self.split}/atomic_numbers", idx, width=1).reshape(-1)

            # Get the positions of the atoms in the structure
            pos = self._read_ragged_sample(reader, f"{self.split}/pos", idx, width=3)

            # Cell is stored as 3 rows of width 3 per structure.
            cell = self._read_ragged_sample(reader, f"{self.split}/cell", idx, width=3).reshape(3, 3)

            # Get all the requested properties
            props = self._read_properties(reader, idx)

        # Make the chemgraph
        data = ChemGraph(
                pos = torch.from_numpy(pos).float() % 1.0,
                cell = torch.from_numpy(cell).unsqueeze(0),
                atomic_numbers = torch.from_numpy(atomic_numbers),
                num_atoms = torch.tensor(natoms),
                num_nodes = torch.tensor(natoms),
                **props
            )
        
        # Apply transforms
        if self.transforms is not None:
            for t in self.transforms:
                data = t(data)

        return data
