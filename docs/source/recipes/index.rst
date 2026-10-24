Recipes
=======

M3L provides experiment workflows under ``recipes/`` for a range of datasets
and tasks. This page describes the common setup and execution pattern; the
individual guides below cover selected recipes. These recipes use GNU Make
stages (``stage0``–``stage5``) and ``.done`` files to track completed steps.

Environment variables
---------------------

Create an ``.envrc`` file in the repository root with paths appropriate
for your environment:

.. code-block:: bash

   export CORPUS_BASE_PATH=/path/to/corpora
   export BEATS_CKPT_PATH=/path/to/beats/checkpoints
   export AIACCEL_JOB_CONFIG="$(pwd)/recipes/clusters/abci3.yaml"

``CORPUS_BASE_PATH`` points to the datasets described in
``recipes/globals.yaml``. ``BEATS_CKPT_PATH`` is used by models
requiring a pretrained BEATs checkpoint. Set ``AIACCEL_JOB_CONFIG``
to the job configuration for your execution environment; the example
uses the included ABCI 3.0 configuration.

After configuring the `direnv shell hook <https://direnv.net/docs/hook.html>`_,
approve the file from the repository root:

.. code-block:: bash

   pixi run direnv allow

Running a recipe
----------------

Install the dependencies with Pixi as described in the
:doc:`User Guide <../user_guide/index>` and obtain the required dataset.
Then enter the recipe directory and create its ``hdf5/`` directory.
For example, to run the DESED recipe:

.. code-block:: bash

   pixi shell
   cd recipes/desed
   mkdir -p hdf5
   make stage5

``make stage5`` runs the preceding stages as needed, from dataset
preparation through training, inference, and evaluation. To execute
independent Make targets in parallel, or choose a different training
configuration, use:

.. code-block:: bash

   make -j stage5
   make -j stage5 train_path="models/sed/dcase_crnn/"

For other recipes, use the corresponding directory and model configuration.
You can also run an individual stage using ``make stageN`` or inspect
the available targets with ``make help``.

Recipe guides
-------------

The following pages describe individual recipes; they are not an exhaustive
list of the workflows under ``recipes/``.

.. toctree::
   :maxdepth: 1

   desed
   urban_sed
