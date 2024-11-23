import torch
from torchaudio.transforms import Spectrogram
import utils.functional as FU
from core.CustomMelScale import CustomMelScale

def pad_sequence(batch):
    # Ensure all items are tensors and have at least 2 dimensions
    batch = [item.t() if item.ndim == 2 else item.unsqueeze(0) for item in batch]
    # Pad sequences
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    # Permute back
    return batch.permute(0, 2, 1)

# def vmi():
#     spectrogram_transform = Spectrogram(n_fft=mel_config.n_fft, hop_length=mel_config.hop_length, power=mel_config.power, center)
#     spectrogram = spectrogram_transform(waveform)    

#     # Initialize CustomMelScale with the filter type and configuration
#     custom_mel_scale = CustomMelScale(
#         n_mels=mel_config.n_mels,
#         sample_rate=mel_config.sample_rate,
#         f_min=mel_config.f_min,
#         f_max=mel_config.f_max,
#         n_stft=mel_config.n_fft // 2 + 1,
#         filter_type=mel_config.filter_type
#     )

#     # # Apply the CustomMelScale transformation to the spectrogram
#     mel_spectrogram = custom_mel_scale(spectrogram)
    
#     return mel_spectrogram, sample_rate, custom_mel_scale



def gen_melspecrogram_common(tensors_or_waveform, n_mels, sample_rate, f_min, f_max, filter, n_fft, hop_length, power, center):
    
    # Spectrogram transformation
    spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=power, center=center)
    spectrogram = spectrogram_transform(tensors_or_waveform)

    # Initialize CustomMelScale with the filter type and configuration
    custom_mel_scale = CustomMelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_stft=n_fft // 2 + 1,
        filter_type=filter
    )

    # Apply the CustomMelScale transformation
    mel_spectrogram = custom_mel_scale(spectrogram)    
    return mel_spectrogram, sample_rate, custom_mel_scale

def preprocess_collate(tensors, targets, n_fft, hop_length, n_mels, sample_rate, f_min, f_max, threshold, filter):

    tensors_padded = pad_sequence(tensors)

    mel_spectrogram,_,_ = gen_melspecrogram_common(tensors_padded, n_mels, sample_rate, f_min, f_max, filter, n_fft, hop_length, power=1.0, center=True)
    
    # # Spectrogram transformation
    # spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0, center=True)
    # spectrogram = spectrogram_transform(tensors_padded)

    # # Initialize CustomMelScale with the filter type and configuration
    # custom_mel_scale = CustomMelScale(
    #     n_mels=n_mels,
    #     sample_rate=sample_rate,
    #     f_min=f_min,
    #     f_max=f_max,
    #     n_stft=n_fft // 2 + 1,
    #     filter_type=filter
    # )

    # # Apply the CustomMelScale transformation
    # mel_spectrogram = custom_mel_scale(spectrogram)
    
    #normalize
    mel_spectrogram_normalized = FU.normalize(mel_spectrogram, normalize_to=1)

    # Time-step adjustment
    delta_t = 1 / sample_rate
    csum = torch.cumsum(mel_spectrogram_normalized, dim=-1) * delta_t

    # Step forward encoding
    base_cums, pos_accum, neg_accum = FU.step_forward_encoding(batch=csum, thr=threshold, neg=False)
    spikes = pos_accum

    # Neuron and spike information
    num_neurons = spikes.shape[2]  # Dimension for neurons
    num_spike_index = spikes.shape[3]

    # Stack targets into a tensor
    targets = torch.tensor(targets)

    return spikes, targets, num_neurons, base_cums
