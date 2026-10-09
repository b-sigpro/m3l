# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

train_path := models/sed/dcase_crnn-beats/
tag := dcase_crnn

inference_name := $(tag)_$(shell date +"%Y-%m-%d-%H-%M-%S")
inference_command := "python -m sbss.nfca.iter.separate $$src_path $$dst_path"

inference_path = results/$(inference_name)

stage1_dependencies = \
	.1_dump_strong_label_names.done \
	.1_convert_strong_groundtruth.done \
	.1_prepare_balanced_sampler_strong.done \
	$(SPLITS:%=.1_convert_wav.%.done)


.1_prepare_balanced_sampler_strong.done: | stage0
	python ./scripts/1_prepare_balanced_sampler_strong.py
	touch $@

stage2_dependencies = \
	$(SPLITS:%=.2_make_audio_strong_hdf5.%.done)

.2_make_audio_strong_hdf5.%.done: | stage1
	$(cmd) mpi $(job_ops) --n_procs=48 jobs/2_make_audio_strong_hdf5/$*.log -- \
		python ./scripts/2_make_audio_strong_hdf5.py $* --parallel
	touch $@