Audio-text packages
===================

Open-vocabulary sound event detection (OV-SED)
----------------------------------------------

Audio encoders
^^^^^^^^^^^^^^

.. currentmodule:: m3l.audiotext.ov_sed.audio_backbone.cnn_beats_encoder

.. autosummary::
    :toctree: generated/

    CNNBEATsEncoder

Text encoders
^^^^^^^^^^^^^

.. currentmodule:: m3l.audiotext.ov_sed.text_backbone.laion_clap_text_encoder

.. autosummary::
    :toctree: generated/

    LaionClapTextEncoder

Models
^^^^^^

.. currentmodule:: m3l.audiotext.ov_sed.models.ovsed

.. autosummary::
    :toctree: generated/

    CrossAttnBlock
    OpenVocabularySEDModel

Tasks
^^^^^

.. currentmodule:: m3l.audiotext.ov_sed.tasks.doom_sed_task

.. autosummary::
    :toctree: generated/

    DOOMSEDTask

Inference pipelines
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: m3l.audiotext.ov_sed.pipelines.detect

.. autosummary::
    :toctree: generated/

    DetectPipeline
