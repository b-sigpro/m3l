# options
SPLITS = train validation eval

sample_rate := 16000
label_resolution := 64

train_path := models/sed/dcase_crnn-beats.weak_ema/
tag := dcase_crnn-beats

inference_name := $(tag)_$(shell date +"%Y-%m-%d-%H-%M-%S")
inference_command := python -m m3l.audio.sed.detect batch $$train_path $$src_path $$dst_path

inference_path = results/$(inference_name)

# constants
STAGES = stage0 stage1 stage2 stage3 stage4 stage5

STRONG_HDF5_SPLITS = train validation
INFERENCE_SPLITS = validation eval

PRINT_VARIABLES = cmd job_ops sample_rate label_resolution train_path tag inference_name inference_command

.PHONY: all clean status $(STAGES) scores $(STAGES:%=skip-%)

include ../globals.mk

# general rules
all: $(STAGES)


# stage0: prepare dataset
stage0: status .stage0.done

.stage0.done: \
	.prepare_dataset.done

.prepare_dataset.done:
	python ./scripts/prepare_dataset.py
	touch $@


# stage1: preprocess
stage1: status .stage1.done

.stage1.done: \
	.summarize_durations.done \
	.dump_ground_truth.done

.summarize_durations.done: .stage0.done
	python ./scripts/summarize_durations.py
	touch $@

.dump_ground_truth.done: .stage0.done
	python ./scripts/dump_ground_truth.py --resolution=64
	touch $@


# stage2: make hdf5
stage2: status \
	.stage2.$(sample_rate)hz_$(label_resolution)ms.done

.stage2.$(sample_rate)hz_$(label_resolution)ms.done: \
	$(STRONG_HDF5_SPLITS:%=.make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms.%.done) \
	.make_hdf5_weak.$(sample_rate)hz.train.done \
	.make_hdf5_unsupervised.$(sample_rate)hz.train.done

.make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms.%.done: .stage1.done
	$(cmd) mpi $(job_ops) --n_procs=40 jobs/make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms/$*.log -- \
		python ./scripts/make_hdf5_strong.py $* --sample_rate=$(sample_rate) --label_resolution=$(label_resolution) --parallel
	touch $@
	
.make_hdf5_weak.$(sample_rate)hz.%.done: .stage1.done
	$(cmd) mpi $(job_ops) --n_procs=40 jobs/make_hdf5_weak.$(sample_rate)hz/$*.log -- \
		python ./scripts/make_hdf5_weak.py $* --sample_rate=$(sample_rate) --parallel
	touch $@
	
.make_hdf5_unsupervised.$(sample_rate)hz.%.done: .stage1.done
	$(cmd) mpi $(job_ops) --n_procs=40 jobs/make_hdf5_unsupervised.$(sample_rate)hz/$*.log -- \
		python ./scripts/make_hdf5_unsupervised.py $* --sample_rate=$(sample_rate) --parallel
	touch $@


# stage3: training
stage3: status $(train_path)/train.done

$(train_path)/.stage3.done: $(train_path)/.train.done

$(train_path)/train.done: $(train_path)/config.yaml .stage2.$(sample_rate)hz_$(label_resolution)ms.done
	# rm -r \
	# 	$(train_path)/train.*
	# 	$(train_path)/merged_config.yaml
	# 	$(train_path)/checkpoints
	# 	$(train_path)/events*
	# 	$(train_path)/hparams.yaml
	$(cmd) train $(job_ops) --n_gpus=8 --walltime=4:0:0 $(job_ops) $(train_path)/train.log -- \
		python -m aiaccel.torch.apps.train $(train_path)/config.yaml
	touch $@


# stage4: inference
stage4: status $(inference_path)/.stage4.done

$(inference_path)/.stage4.done: \
	$(INFERENCE_SPLITS:%=$(inference_path)/.inference.%.done)

$(inference_path)/.inference.%.done: $(train_path)/.stage3.done
	n_tasks=$(shell find audio/$*/ -name "*.wav" | wc -l) && \
	train_path=$(train_path) && \
	src_path=audio/$* && \
	dst_path=$(inference_path)/$*/scores && \
	$(cmd) gpu --n_tasks=$$n_tasks --n_tasks_per_job=512 $(job_ops) $(inference_path)/jobs/inference.$*.log -- \
		$(inference_command)
	touch $@

# stage5: evaluation
stage5: status $(inference_path)/.stage5.done scores

$(inference_path)/.stage5.done: \
	$(INFERENCE_SPLITS:%=$(inference_path)/.evaluate_sed_scores.%.done)

$(inference_path)/.evaluate_sed_scores.%.done: $(inference_path)/.stage4.done
	python ./scripts/evaluate_sed_scores.py $* $(inference_path) --label_resolution=$(label_resolution)
	touch $@

scores: $(inference_path)/.stage5.done
	python ./scripts/summarize_scores.py $(inference_path)


# clean
clean:
	-rm -r \
		./annotation \
		./audio \
		./metadata/label_mat \
		./metadata/durations* \
		./hdf5/* \
		./jobs

	-rm .*.done
