# options
cmd ?= ../clusters/abci3.py
job_ops ?= 

SPLITS = train eval

train_path ?= models/avc/cnn14_vgg/

# stage definitions
STAGES = stage0 stage1 stage2

.PHONY: all clean $(STAGES)

all: $(STAGES)


# general rules
.%.done:
	python scripts/$(*).py

.stage%.done:
	touch $@


# stage0
stage0:

# stage1
stage1: .stage1.done

.stage1.done: \
	.dump_strong_label_names.done \
	$(SPLITS:%=.convert_wav.%.done) \
	$(SPLITS:%=.convert_mp4.%.done) \

.dump_strong_label_names.done: 
	python ./scripts/dump_strong_label_names.py
	touch $@

.convert_wav.%.done:
	n_tasks=$(shell find $*/audios -name "*.wav" | wc -l) && \
	$(cmd) cpu $(job_ops) ./jobs/convert_wav/$*.log -- python ./scripts/convert_wav.py $*
	touch $@

.convert_mp4.%.done:
	n_tasks=$(shell find $*/videos -name "*.mp4" | wc -l) && \
	$(cmd) cpu $(job_ops) ./jobs/convert_mp4/$*.log -- python ./scripts/convert_mp4.py $*
	touch $@

# stage2
stage2: .stage2.done

.stage2.done: .make_hdf5.done
	$(SPLITS:%=.make_hdf5.%.done)

.make_hdf5.%.done: .stage1.done
	$(cmd) mpi $(job_ops) --n_procs=48 jobs/make_hdf5/$*.log -- python ./scripts/make_hdf5.py $*

# stage3
stage3: $(train_path)/train.done

$(train_path)/train.done: $(train_path)/config.yaml
	$(cmd) train $(job_ops) --n_gpus=8 $(train_path)/train.log -- \
		python -m aiaccel.torch.apps.train $(train_path)/config.yaml

# clean
clean:
	-rm \
		./metadata \
		./metadata_strong \
		./audios \
		./videos \
		./hdf5/* \
		./jobs/ \

	-rm .*.done