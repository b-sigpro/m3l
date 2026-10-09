# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

train_path := models/sed/dcase_crnn/
tag := dcase_crnn

inference_name := $(tag)_$(shell date +"%Y-%m-%d-%H-%M-%S")
inference_command := python -m m3l.audio.sed.detect batch $$train_path $$src_path $$dst_path

inference_path = results/$(inference_name)

# options
sample_rate := 16000
label_resolution := 64

# constants
SPLITS = validate test
TRAIN_SPLITS = train validate
INFERENCE_SPLITS = $(SPLITS)

PRINT_VARIABLES = cmd job_ops sample_rate label_resolution train_path tag inference_name inference_command

# stage0: prepare dataset
stage0_dependencies = \
	.0_prepare_dataset.done \
	hdf5/

.0_prepare_dataset.done:
	python ./scripts/0_prepare_dataset.py
	touch $@

# stage1: preprocess
stage1_dependencies = \
	.1_summarize_durations.done \
	.1_dump_ground_truth.done

.1_summarize_durations.done: | stage0
	python ./scripts/1_summarize_durations.py
	touch $@

.1_dump_ground_truth.done: | stage0
	python ./scripts/1_dump_ground_truth.py
	touch $@


# stage2: make hdf5
stage2_dependencies = \
	$(TRAIN_SPLITS:%=.make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms.%.done)

.make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms.%.done: | stage1
	$(cmd) mpi $(job_ops) --n_procs=40 jobs/make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms/$*.log -- \
		python ./scripts/2_make_hdf5_strong.py $* --sample_rate=$(sample_rate) --label_resolution=$(label_resolution) --parallel
	touch $@


# stage4: inference
stage4_dependencies = \
	$(INFERENCE_SPLITS:%=$(inference_path)/.inference.%.done)

$(inference_path)/.inference.%.done: | stage3
	n_tasks=$(shell find audio/$*/ -name "*.wav" | wc -l) && \
	train_path=$(train_path) && \
	src_path=audio/$* && \
	dst_path=$(inference_path)/$*/scores && \
	$(cmd) gpu --n_tasks=$$n_tasks $(job_ops) $(inference_path)/jobs/inference.$*.log -- \
		$(inference_command)
	touch $@

# stage5: evaluation
stage5: | .WAIT scores

stage5_dependencies = \
	$(INFERENCE_SPLITS:%=$(inference_path)/.evaluate_sed_scores.%.done)

$(inference_path)/.evaluate_sed_scores.%.done: | stage4
	python ./scripts/5_evaluate_sed_scores.py $* $(inference_path) --label_resolution=$(label_resolution)
	touch $@

scores:
	python ./scripts/summarize_scores.py $(inference_path) --splits validate test

# clean
clean:
	git clean -xdfi

include ../globals.mk
