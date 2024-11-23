import utils.preprocess_collate as PF
# Inputs
spikes, targets, num_neurons, base_cums = PF.preprocess_collate(
    tensors=tensors,
    targets=targets,
    n_fft=512,
    hop_length=256,
    n_mels=64,
    sample_rate=16000,
    f_min=0.0,
    f_max=8000.0,
    threshold=0.5,
    filter="custom"
)
