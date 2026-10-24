DESED
=====

The ``recipes/desed/`` directory provides an end-to-end sound event
detection recipe using DESED. It prepares the data, generates HDF5 files,
trains a model, and evaluates its predictions.

See the :doc:`Recipes setup <index>` for environment variables and
Make commands.

Dataset and models
------------------

Obtain DESED and the DCASE 2021 public evaluation data. Their locations
are defined relative to ``CORPUS_BASE_PATH`` in ``recipes/globals.yaml``:

- ``desed/data/dcase/dataset/`` for DESED.
- ``dcase2021_public_eval/`` for the public evaluation data.

The default training configuration is
``models/sed/dcase_crnn-beats.weak_ema/``, which requires
``BEATs_iter3_plus_AS2M.pt`` in ``BEATS_CKPT_PATH``. Other
configurations include ``dcase_crnn-beats``, ``dcase_crnn.weak_ema``,
and ``dcase_crnn`` (without BEATs). Select a configuration with
``train_path`` when invoking Make.

Stage-by-stage guide
--------------------

Stage 0: dataset preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``make stage0`` runs ``scripts/0_prepare_dataset.py`` to link the
audio and annotations and prepare ``annotation/eval.tsv``. It does not
download the datasets.

Stage 1: preprocessing
~~~~~~~~~~~~~~~~~~~~~~

``make stage1`` generates duration metadata for ``validation`` and
``eval``, and frame-level validation labels.

Stage 2: HDF5 generation
~~~~~~~~~~~~~~~~~~~~~~~~

``make stage2`` creates strongly labeled HDF5 files for ``train``,
``synthetic21_validation``, and ``validation``, plus weakly labeled
and unlabeled training datasets under ``hdf5/``. By default, audio is
sampled at 16 kHz with 64 ms label resolution.

Stage 3: training
~~~~~~~~~~~~~~~~~

``make stage3`` trains the selected model using
``$(train_path)/config.yaml``. The shared model settings are in
`recipes/desed/models/sed/base.yaml <https://github.com/b-sigpro/m3l/blob/main/recipes/desed/models/sed/base.yaml>`_.
The Makefile asks before deleting existing training artifacts.

Stage 4: inference
~~~~~~~~~~~~~~~~~~

``make stage4`` runs sound event detection on ``validation`` and
``eval`` and saves predictions in
``results/<inference_name>/<split>/scores/``.

Stage 5: evaluation
~~~~~~~~~~~~~~~~~~~

``make stage5`` computes PSDS1, PSDS2, event-based F1, and AUROC
for ``validation`` and ``eval``. The results are stored in
``results/<inference_name>/<split>/sed_scores.pkl`` and summarized
in the terminal.

When running inference and evaluation in separate Make invocations,
specify the same ``inference_name`` for Stages 4 and 5.
