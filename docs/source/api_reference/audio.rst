Audio packages
==============

Common
------

Datasets
^^^^^^^^

.. currentmodule:: m3l.audio.common.datasets

.. autosummary::
    :toctree: generated/

    ClipLabeledHDF5Dataset
    FrameLabeledHDF5Dataset
    UnlabeledHDF5Dataset

Postprocessors
^^^^^^^^^^^^^^

.. currentmodule:: m3l.audio.common.postprocessors

.. autosummary::
    :toctree: generated/

    MedianFilter

Preprocessors
^^^^^^^^^^^^^

.. currentmodule:: m3l.audio.common.preprocessors

.. autosummary::
    :toctree: generated/

    BatchNormalize
    Clamp
    GaussianNoise
    Mixup
    Normalize
    Preprocessor
    Scale
    SpecAugment
    TimeRoll
    TimeWarp

Encoders
^^^^^^^^

.. currentmodule:: m3l.audio.common.encoders

.. autosummary::
    :toctree: generated/

    BEATsModel
    CNN14
    CRNN
    

Sound event detection (SED)
---------------------------

Callbacks
^^^^^^^^^

.. currentmodule:: m3l.audio.sed.callbacks

.. autosummary::
    :toctree: generated/

    VisualizerCallback

Heads
^^^^^

.. currentmodule:: m3l.audio.sed.heads

.. autosummary::
    :toctree: generated/

    AttentionHead
    LinearHead

Models
^^^^^^

.. currentmodule:: m3l.audio.sed.models

.. autosummary::
    :toctree: generated/

    StrongSEDModel
    StrongWeakSEDModel

Tasks
^^^^^

.. currentmodule:: m3l.audio.sed.tasks

.. autosummary::
    :toctree: generated/

    SEDTask
    WeakEMASEDTask

Tagging
-------

Callbacks
^^^^^^^^^

.. currentmodule:: m3l.audio.tagging.callbacks

.. autosummary::
    :toctree: generated/

    VisualizerCallback

Models
^^^^^^

.. currentmodule:: m3l.audio.tagging.models

.. autosummary::
    :toctree: generated/

    CNN14
    HTSAT
    SwinTransformerBlock
    TaggingModel

Tasks
^^^^^

.. currentmodule:: m3l.audio.tagging.tasks

.. autosummary::
    :toctree: generated/

    infaug_tagging_task.TaggingTask
    tagging_task.TaggingTask
