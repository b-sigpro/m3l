train_path := models/avc/cnn14/
tag := cnn14

inference_name := $(tag)_$(shell date +"%Y-%m-%d-%H-%M-%S")
inference_command := "python -m sbss.nfca.iter.separate $$src_path $$dst_path"

inference_path = results/$(inference_name)


# stage 1
.stage1.done: \
	.dump_strong_label_names.done \
	$(SPLITS:%=.convert_wav.%.done) \
	$(SPLITS:%=.convert_mp4.%.done)


# stage 2
stage2: .stage2.done

.stage2.done: $(SPLITS:%=.make_audiovisual_unsupervised_hdf5.%.done)

.make_audiovisual_unsupervised_hdf5.%.done: .stage1.done
	$(cmd) mpi $(job_ops) --n_procs=48 jobs/make_audiovisual_unsupervised_hdf5/$*.log -- \
		python ./scripts/make_audiovisual_unsupervised_hdf5.py $* --parallel