# options
cmd ?= ../clusters/abci3.py
job_ops ?= 

sample_rate ?= 16000
label_resolution ?= 64

train_path ?= models/sed/dcase_crnn-beats.weak_ema/

# stage definitions
STAGES = stage0 stage1 stage2 stage3

.PHONY: all clean $(STAGES)

all: $(STAGES)


# general rules
.stage%.done:
	touch $@


# stage0
stage0: .stage0.done

.stage0.done: \
	.prepare_dataset.done

.prepare_dataset.done:
	python ./scripts/prepare_dataset.py
	touch $@

# stage1
stage1: .stage1.done

.stage1.done: \
	.summarize_durations.done \
	.dump_ground_truth.done

.summarize_durations.done: .stage0.done
	python ./scripts/summarize_durations.py
	touch $@

.dump_ground_truth.done: .stage0.done
	python ./scripts/dump_ground_truth.py
	touch $@

# stage2
stage2: \
	.stage2.$(sample_rate)hz_$(label_resolution)ms.done

STRONG_HDF5_SPLITS = train validation
.stage2.$(sample_rate)hz_$(label_resolution)ms.done: \
	$(STRONG_HDF5_SPLITS:%=.make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms.%.done) \
	.make_hdf5_weak.$(sample_rate)hz.train.done \
	.make_hdf5_unsupervised.$(sample_rate)hz.train.done

.make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms.%.done: .stage1.done
	$(cmd) mpi $(job_ops) --n_procs=40 jobs/make_hdf5_strong.$(sample_rate)hz_$(label_resolution)ms/$*.log -- \
		python ./scripts/make_hdf5_strong.py $* --sample_rate=$(sample_rate) --label_resolution=$(label_resolution)
	
.make_hdf5_weak.$(sample_rate)hz.%.done: .stage1.done
	$(cmd) mpi $(job_ops) --n_procs=40 jobs/make_hdf5_weak.$(sample_rate)hz/$*.log -- \
		python ./scripts/make_hdf5_weak.py $* --sample_rate=$(sample_rate)
	
.make_hdf5_unsupervised.$(sample_rate)hz.%.done: .stage1.done
	$(cmd) mpi $(job_ops) --n_procs=40 jobs/make_hdf5_unsupervised.$(sample_rate)hz/$*.log -- \
		python ./scripts/make_hdf5_unsupervised.py $* --sample_rate=$(sample_rate)


# stage3
stage3: $(train_path)/train.done

$(train_path)/train.done: $(train_path)/config.yaml .stage2.done
	$(cmd) train $(job_ops) --n_gpus=8 $(train_path)/train.log -- python -m aiaccel.torch.apps.train $(train_path)/config.yaml


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
