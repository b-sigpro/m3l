URBAN-SED
=========

The ``recipes/urban-sed/`` directory provides an end-to-end sound
event detection recipe using URBAN-SED, from data preparation to training,
inference, and evaluation.

See the :doc:`Recipes setup <index>` for environment variables and
Make commands.

Dataset and model
-----------------

Obtain URBAN-SED v2.0.0 and place it under
``<CORPUS_BASE_PATH>/urban-sed/URBAN-SED_v2.0.0/``. The recipe expects
``audio/`` and ``annotations/`` directories with ``train``,
``validate``, and ``test`` splits.

The default training configuration is
`recipes/urban-sed/models/sed/dcase_crnn/config.yaml <https://github.com/b-sigpro/m3l/blob/main/recipes/urban-sed/models/sed/dcase_crnn/config.yaml>`_.

Stage-by-stage guide
--------------------

Stage 0: dataset preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``make stage0`` links the audio and annotations into the recipe
directory. It does not download the dataset.

Stage 1: preprocessing
~~~~~~~~~~~~~~~~~~~~~~

``make stage1`` generates event ground truth for ``train``,
``validate``, and ``test``, and duration metadata for
``validate`` and ``test``.

Stage 2: HDF5 generation
~~~~~~~~~~~~~~~~~~~~~~~~

``make stage2`` creates strongly labeled HDF5 datasets for
``train`` and ``validate`` under ``hdf5/``. The defaults are
16 kHz audio and 64 ms label resolution.

Stage 3: training
~~~~~~~~~~~~~~~~~

``make stage3`` trains the CRNN from
``models/sed/dcase_crnn/config.yaml``. The Makefile asks before
deleting existing training artifacts.

Stage 4: inference
~~~~~~~~~~~~~~~~~~

``make stage4`` runs sound event detection on ``validate``
and ``test``, saving predictions in
``results/<inference_name>/<split>/scores/``.

Stage 5: evaluation
~~~~~~~~~~~~~~~~~~~

``make stage5`` computes PSDS1, PSDS2, event-based F1, and AUROC
for ``validate`` and ``test``. The results are stored in
``results/<inference_name>/<split>/sed_scores.pkl`` and summarized
in the terminal.

When running inference and evaluation in separate Make invocations,
specify the same ``inference_name`` for Stages 4 and 5.
