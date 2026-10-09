# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# options
mode = sed

SPLITS = train eval

PRINT_VARIABLES = cmd job_ops mode train_path tag inference_name inference_command

# stage0
stage0_dependencies = \
	.0_prepare_dataset.done \
	hdf5/

.0_prepare_dataset.done:
	python ./scripts/0_prepare_dataset.py
	touch $@

# stage1
.1_dump_strong_label_names.done: | stage0
	python ./scripts/1_dump_strong_label_names.py
	touch $@
	
.1_convert_strong_groundtruth.done: | stage0
	python ./scripts/1_convert_strong_groundtruth.py
	touch $@

.1_convert_wav.%.done: | stage0
	n_tasks=$(shell find raw/audios/*$** -name "*.wav" | wc -l) && \
	echo $$n_tasks && \
	$(cmd) cpu $(job_ops) --n_tasks=$$n_tasks --n_tasks_per_proc=16384 --n_proc=1 ./jobs/1_convert_wav/$*.log -- \
		python ./scripts/1_convert_wav.py $*
	touch $@

.1_convert_mp4.%.done: | stage0
	n_tasks=$(shell find $*/videos -name "*.mp4" | wc -l) && \
	$(cmd) cpu $(job_ops) ./jobs/1_convert_mp4/$*.log -- python ./scripts/1_convert_mp4.py $*
	touch $@

stage3_dependencies := $(train_path)/.train.done

# clean
clean:
	git clean -xdfi

include $(mode).mk
include ../globals.mk
