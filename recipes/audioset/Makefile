# options
mode = sed

SPLITS = train eval

# stage definitions
STAGES = stage0 stage1 stage2

PRINT_VARIABLES = cmd job_ops mode train_path tag inference_name inference_command

.PHONY: all clean status $(STAGES) scores $(STAGES:%=skip-%)

include ../globals.mk
include $(mode).mk

# general rules
all: $(STAGES)

# stage0
stage0: status .stage0.done

.stage0.done: \
	.prepare_dataset.done


.prepare_dataset.done:
	python ./scripts/prepare_dataset.py
	touch $@


# stage1
stage1: .stage1.done

.dump_strong_label_names.done: .stage0.done
	python ./scripts/dump_strong_label_names.py
	touch $@
	
.convert_strong_groundtruth.done: .stage0.done
	python ./scripts/convert_strong_groundtruth.py
	touch $@

.convert_wav.%.done: .stage0.done
	n_tasks=$(shell find raw/audios/*$** -name "*.wav" | wc -l) && \
	echo $$n_tasks && \
	$(cmd) cpu $(job_ops) --n_tasks=$$n_tasks --n_tasks_per_job=16384 --n_procs_per_job=1 ./jobs/convert_wav/$*.log -- \
		python ./scripts/convert_wav.py $*
	touch $@

.convert_mp4.%.done: .stage0.done
	n_tasks=$(shell find $*/videos -name "*.mp4" | wc -l) && \
	$(cmd) cpu $(job_ops) ./jobs/convert_mp4/$*.log -- python ./scripts/convert_mp4.py $*
	touch $@

# stage3
stage3: status $(train_path)/train.done

$(train_path)/.stage3.done: $(train_path)/.train.done

$(train_path)/train.done: $(train_path)/config.yaml .stage2.done
	$(cmd) train $(job_ops) --n_gpus=8 --walltime=12:0:0 $(job_ops) $(train_path)/train.log -- \
			python -m aiaccel.torch.apps.train $(train_path)/config.yaml
	touch $@

# clean
clean:
	-rm -r \
		./metadata \
		./raw \
		./train \
		./eval \
		./hdf5/* \
		./jobs \

	-rm .*.done