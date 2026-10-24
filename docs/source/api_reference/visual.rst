Visual packages
===============

Common
------

Encoders
^^^^^^^^

.. currentmodule:: m3l.visual.common.encoders

.. autosummary::
    :toctree: generated/

    TorchvisionResNet50
    TorchvisionResNet50MultiScale
    VGGLike8

Transforms
^^^^^^^^^^

.. currentmodule:: m3l.visual.common.transforms

.. autosummary::
    :toctree: generated/

    Compose
    ConvertBoundingBoxesFormat
    FilterCrowdBBox
    LabelsGetter
    NormalizeBoundingBoxes
    RandomChoice
    RangeRandomCrop

Metrics
^^^^^^^

.. currentmodule:: m3l.visual.common.metrics.mean_ap

.. autosummary::
    :toctree: generated/

    MeanAveragePrecision

Batching
^^^^^^^^

.. currentmodule:: m3l.visual.common.nest_collator

.. autosummary::
    :toctree: generated/

    NestedTensorCollator
