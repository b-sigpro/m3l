# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

train_path := models/sed/dcase_crnn/
tag := dcase_crnn

inference_name := $(tag)_$(shell date +"%Y-%m-%d-%H-%M-%S")
inference_command := python -m m3l.audio.sed.detect batch $$train_path $$src_path $$dst_path

inference_path = results/$(inference_name)

include ../globals.mk

# options
sample_rate := 16000
label_resolution := 64

# constants
SPLITS = validate test
TRAIN_SPLITS = train validate
INFERENCE_SPLITS = $(SPLITS)

PRINT_VARIABLES = cmd job_ops sample_rate label_resolution train_path tag inference_name inference_command

# stage0: prepare dataset
.stage0.done: | \
	.prepare_dataset.done

.prepare_dataset.done:
	python ./scripts/prepare_dataset.py
	touch $@


# stage1: preprocess
.stage1.done: | \
	.summarize_durations.done \
	.dump_ground_truth.done

.summarize_durations.done: | .stage0.done
	python ./scripts/summarize_durations.py
	touch $@

.dump_ground_truth.done: | .stage0.done
	python ./scripts/dump_ground_truth.py
	touch $@


# stage2: make hdf5
.stage2.done: | \
	$(TRAIN_SPLITS:%=.make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms.%.done)

.make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms.%.done: | .stage1.done
	$(cmd) mpi $(job_ops) --n_procs=40 jobs/make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms/$*.log -- \
		python ./scripts/make_hdf5_strong.py $* --sample_rate=$(sample_rate) --label_resolution=$(label_resolution) --parallel
	touch $@


# stage4: inference
$(inference_path)/.stage4.done: | \
	$(INFERENCE_SPLITS:%=$(inference_path)/.inference.%.done)

$(inference_path)/.inference.%.done: | $(train_path)/.stage3.done
	n_tasks=$(shell find audio/$*/ -name "*.wav" | wc -l) && \
	train_path=$(train_path) && \
	src_path=audio/$* && \
	dst_path=$(inference_path)/$*/scores && \
	$(cmd) gpu --n_tasks=$$n_tasks --n_tasks_per_job=512 $(job_ops) $(inference_path)/jobs/inference.$*.log -- \
		$(inference_command)
	touch $@

# stage5: evaluation
stage5: status | $(inference_path)/.stage5.done scores

$(inference_path)/.stage5.done: | \
	$(INFERENCE_SPLITS:%=$(inference_path)/.evaluate_sed_scores.%.done)

$(inference_path)/.evaluate_sed_scores.%.done: | $(inference_path)/.stage4.done
	python ./scripts/evaluate_sed_scores.py $* $(inference_path) --label_resolution=$(label_resolution)
	touch $@

scores: | $(inference_path)/.stage5.done
	python ./scripts/summarize_scores.py $(inference_path)


# clean
clean:
	-rm -r \
		./annotation \
		./audio \
		./metadata \
		./jobs

	-rm .*.done
