# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

WEAK_SPLITS := train eval
sample_rate := 16000

train_walltime := 48:0:0

.stage1.done: | \
	$(SPLITS:%=.convert_wav.%.done) \
	.prepare_balanced_sampler.done
	touch $@

.prepare_balanced_sampler.done: | .stage0.done
	python ./scripts/prepare_balanced_sampler.py
	touch $@

.stage2.done: | $(WEAK_SPLITS:%=.make_audio_weak_hdf5.%.done)
	touch $@

.make_audio_weak_hdf5.%.done: | .stage1.done
	mkdir -p ./hdf5 jobs/make_audio_weak_hdf5
	$(cmd) mpi $(job_ops) --walltime=48:0:0 --n_nodes=4 --n_procs=192 jobs/make_audio_weak_hdf5/$*.log -- \
		python ./scripts/make_audio_weak_hdf5.py $* --sample_rate=$(sample_rate) --parallel
	touch $@
