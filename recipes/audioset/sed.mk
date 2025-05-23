train_path := models/sed/dcase_crnn-beats/
tag := dcase_crnn

inference_name := $(tag)_$(shell date +"%Y-%m-%d-%H-%M-%S")
inference_command := "python -m sbss.nfca.iter.separate $$src_path $$dst_path"

inference_path = results/$(inference_name)

# stage 1
.stage1.done: \
	.dump_strong_label_names.done \
	.convert_strong_groundtruth.done \
	$(SPLITS:%=.convert_wav.%.done)


# stage 2
stage2: .stage2.done

.stage2.done: $(SPLITS:%=.make_audio_strong_hdf5.%.done)

.make_audio_strong_hdf5.%.done: .stage1.done
	$(cmd) mpi $(job_ops) --n_procs=48 jobs/make_audio_strong_hdf5/$*.log -- \
		python ./scripts/make_audio_strong_hdf5.py $* --parallel
	touch $@