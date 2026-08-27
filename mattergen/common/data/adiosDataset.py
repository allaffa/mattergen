import torch
from torch.utils.data import Dataset
from mattergen.common.data.chemgraph import ChemGraph
import adios2
import numpy as np
import re


def _parse_adios_shape(value, variable):
    values = re.findall(r"\d+", value) if isinstance(value, str) else value
    shape = tuple(int(item) for item in values)
    if not shape:
        raise ValueError(f"ADIOS variable {variable!r} has no global shape")
    return shape

# this is terrible but trying to load twice will make it work...
try:    
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset
except:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset


class HydraGNNAdiosCrystalDataset(AdiosDataset):
    """Adapt HydraGNN's persistent ADIOS reader to MatterGen ChemGraphs."""

    def __init__(self, *args, transforms=None,properties=None, **kwargs):
        self.transforms = transforms
        self.property_names = set(properties or [])

        super().__init__(*args, **kwargs)
        self.node_count_variable = f"{self.label}/natoms"
        atomic_numbers_variable = f"{self.label}/atomic_numbers"
        variables = self.f.available_variables()
        atomic_shape = _parse_adios_shape(
            variables[atomic_numbers_variable]["Shape"], atomic_numbers_variable
        )
        non_unit_extents = [extent for extent in atomic_shape if extent != 1]
        if len(non_unit_extents) != 1:
            raise ValueError(
                f"Cannot infer total atoms from {atomic_numbers_variable!r} "
                f"with shape {atomic_shape}"
            )
        # Lets the streaming sampler derive average atoms/sample from ADIOS
        # global metadata without reading a billion-entry natoms array.
        self.total_node_count = int(non_unit_extents[0])

    def read_node_counts_range(self, start, count):
        """Read one contiguous natoms range through the persistent reader."""

        return np.asarray(
            self.f.read(
                self.node_count_variable,
                [int(start)],
                [int(count)],
            ),
            dtype=np.int64,
        ).reshape(-1)

    def __getitem__(self, idx) -> ChemGraph:

        # just use the parent class get method
        object = super().get(idx)

        # Get number of atoms in the structure
        natoms = object.atomic_numbers.shape[0]

        # Get the atomic numbers of the atoms in the structure 
        atomic_numbers = object.atomic_numbers.reshape(-1)

        # Get the positions of the atoms in the structure
        pos = object.pos

        # Cell is stored as 3 rows of width 3 per structure.
        cell = object.cell.reshape(3,3)

        # Get all the requested properties - leave for later
        props = {}

        # Read in forces and compute force_rms if requested
        if "force_rms" in self.property_names:
            forces = torch.as_tensor(object.forces, dtype=torch.float32).reshape(natoms, 3)
            force_rms = torch.sqrt(torch.sum(forces.square(), dim=-1).mean())
            props["force_rms"] = force_rms.clamp_min(1e-8).reshape(1)

        # Make the chemgraph
        data = ChemGraph(
                pos = pos % 1.0,
                cell = cell.unsqueeze(0),
                atomic_numbers = atomic_numbers,
                num_atoms = natoms,
                num_nodes = natoms,
                **props
            )

        # Apply transforms
        if self.transforms is not None:
            for t in self.transforms:
                data = t(data)

        # spit it back
        return(data)
    

    


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
