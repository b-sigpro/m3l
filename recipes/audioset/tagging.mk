# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

train_path := models/tagging/cnn14/

WEAK_SPLITS := train eval
sample_rate := 16000

stage1_dependencies = \
	$(SPLITS:%=.1_convert_wav.%.done) \
	.1_dump_weak_label_names.done \
	.1_prepare_balanced_sampler.done

.1_prepare_balanced_sampler.done: | stage0
	python ./scripts/1_prepare_balanced_sampler.py
	touch $@

.1_dump_weak_label_names.done: | stage0
	python ./scripts/1_dump_weak_label_names.py
	touch $@

stage2_dependencies = \
	$(WEAK_SPLITS:%=.2_make_audio_weak_hdf5.%.done)
	
.2_make_audio_weak_hdf5.%.done: | stage1
	$(cmd) mpi $(job_ops) --walltime=20:0:0 --n_nodes=1 --n_procs=96 jobs/2_make_audio_weak_hdf5/$*.log -- \
			python ./scripts/2_make_audio_weak_hdf5.py $* --sample_rate=$(sample_rate) --parallel
	touch $@
