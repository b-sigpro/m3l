<div align="center"><img src="https://raw.githubusercontent.com/b-sigpro/m3l/master/docs/image/logo_m3l.png" height="140"/></div>

<div align="center">
    <img src="https://img.shields.io/github/license/b-sigpro/m3l.svg" />
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
    <a href="https://m3l-toolkit.readthedocs.io/en/latest/index.html">
    <a href="https://pixi.sh/">
        <img src="https://img.shields.io/badge/Pixi-managed-yellow" alt="Pixi" />
    </a>
    <a href="https://m3l-toolkit.readthedocs.io/en/latest/index.html">
        <img src="https://app.readthedocs.org/projects/m3l-toolkit/badge/?version=latest" />
    </a>
</div>

# M3L: Multi-Modal Machine Listening Toolkit

M3L is a research toolkit for multi-modal machine listening, including audio representation learning, sound event detection, and reproducible experiment recipes.

## Key features

- **Reproducible recipes** – End-to-end pipelines document every stage (data prep, training, inference, evaluation) for reproducible studies.
- **HPC ready** – Recipes and Makefiles are tuned for ABCI, TSUBAME, and other clusters, yet remain runnable on a single workstation.
- **Highly modular** – Lightning tasks, common datasets, models, and utilities are built to swap components and run ablations with minimal friction.

## Getting started

For inference or use as a Python dependency, install M3L from GitHub:

```bash
pip install git+https://github.com/b-sigpro/m3l
```

For development, use [Pixi](https://pixi.sh/) to manage the project environment.

```bash
git clone git@github.com:b-sigpro/m3l.git
cd m3l
pixi install
pixi run pre-commit install
```

> [!NOTE]
> The Pixi workspace currently supports Linux (`linux-64`), with a GPU-oriented default environment and a CPU environment for CI.

## Recipes

Example experiment recipes are available under `recipes/`:

- **[DESED](recipes/desed/)** – Sound event detection using the DESED dataset.
- **[URBAN-SED](recipes/urban-sed/)** – Sound event detection using synthetic urban soundscapes.
- **[AudioSet](recipes/audioset/)** – Audio tagging, sound event detection, and audio-visual learning.

Dataset preparation and experiment execution require the corresponding datasets and recipe-specific configuration.

> [!NOTE]
> These recipes are currently under maintenance.

## License

M3L is licensed under the [MIT License](LICENSE), except for the following files and directories, which are subject to their own licenses and copyright notices:

- [`m3l/common/nn/transformer.py`](m3l/common/nn/transformer.py) – `TransformerEncoderLayer` and `TransformerDecoderLayer` adapted from [PyTorch](https://github.com/pytorch/pytorch), licensed under [MIT AND BSD-3-Clause](LICENSE-PYTORCH).
- [`m3l/third/beats/`](m3l/third/beats/) – [BEATs](https://github.com/microsoft/unilm/blob/master/beats/README.md), licensed under its own [MIT](LICENSE-BEATS).

Please refer to the linked license files before reusing or redistributing these components.

## Acknowledgement

- Part of this software was developed in a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
- Part of this software was developed by using ABCI 3.0 provided by AIST and AIST Solutions.
- Part of this software was developed by using the TSUBAME4.0 supercomputer at Institute of Science Tokyo.
