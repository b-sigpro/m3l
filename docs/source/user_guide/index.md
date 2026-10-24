# User Guide

## Installation

### Install from GitHub (inference)

To use M3L for inference or as a Python dependency, install it directly from GitHub:

```bash
python -m pip install git+https://github.com/b-sigpro/m3l
```

### Install with Pixi (training and development)

For training experiments or developing M3L:

1. Install [Pixi](https://pixi.prefix.dev/latest/) if it is not already available.
2. Clone the repository:

   ```bash
   git clone git@github.com:b-sigpro/m3l.git
   cd m3l
   ```

3. Install the dependencies:

   ```bash
   pixi install
   ```

Pixi provides GNU Make (4.4 or later), direnv, and the other tools used by
the recipes. The workspace targets Linux (`linux-64`); its default
environment includes GPU dependencies.

For development, you can also install the Git hooks:

```bash
pixi run pre-commit install
```

### MPI configuration

The default Pixi environment uses the `system-mpi` feature. The
[`pixi.toml`](https://github.com/b-sigpro/m3l/blob/main/pixi.toml)
configuration requests OpenMPI 4.1 and MPI-enabled builds of HDF5 and h5py.
Check that your system's MPI installation is compatible with these dependencies,
especially on an HPC cluster.

## Recipes

To prepare datasets, train models, run inference, or evaluate results, use the
workflows under `recipes/`. See the {doc}`Recipes <../recipes/index>`
documentation for setup and usage.

## Recommended reading

M3L uses `aiaccel` for experiment configuration, job execution, and
PyTorch Lightning integration. For those shared components, refer to the
[aiaccel user guide](https://aiaccel.readthedocs.io/en/latest/user_guide/).
