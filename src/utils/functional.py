import torch as torch
from torchaudio.transforms import Spectrogram
from core.CustomMelScale import CustomMelScale
import torchaudio.transforms as T

def get_mel_spectrogram(audio_sample, mel_config):
    """
    Compute the mel spectrogram using the filter type specified in mel_config.
    """
    waveform, sample_rate = audio_sample.load_waveform()

    #debug:
    # mel_spectrogram = mel_config.debug_transform(waveform)        
    
    # Apply Spectrogram transformation first, then use CustomMelScale for filtering
    spectrogram_transform = Spectrogram(n_fft=mel_config.n_fft, hop_length=mel_config.hop_length, power=mel_config.power)
    spectrogram = spectrogram_transform(waveform)    

    # Initialize CustomMelScale with the filter type and configuration
    custom_mel_scale = CustomMelScale(
        n_mels=mel_config.n_mels,
        sample_rate=mel_config.sample_rate,
        f_min=mel_config.f_min,
        f_max=mel_config.f_max,
        n_stft=mel_config.n_fft // 2 + 1,
        filter_type=mel_config.filter_type
    )

    # # Apply the CustomMelScale transformation to the spectrogram
    mel_spectrogram = custom_mel_scale(spectrogram)
    
    return mel_spectrogram, sample_rate, custom_mel_scale


def find_max_interval(spikes, channel, time_step):
    """
    Finds the interval in the specified channel with the maximum number of 1's.
    
    Args:
        spikes (torch.Tensor): Tensor with shape (num_channels, num_time_steps) containing binary values.
        channel (int): The channel number where to perform the search.
        time_step (int): The size of the interval to consider.

    Returns:
        tuple: (start_index, end_index, max_count) where:
            - start_index is the starting time step of the interval with max 1's.
            - end_index is the end time step of this interval.
            - max_count is the count of 1's in this interval.
    """
    # Select the specified channel
    channel = channel-1# channel 1 is zero in tensor
    channel_data = spikes[channel]
    
    max_count = 0
    max_interval = (0, time_step)  # Initializing with the first interval
    
    # Slide the window across the time steps
    for start in range(len(channel_data) - time_step + 1):
        end = start + time_step
        current_count = channel_data[start:end].sum().item()  # Sum 1's in the current window

        if current_count > max_count:
            max_count = current_count
            max_interval = (start, end)

    return max_interval[0], max_interval[1], max_count

def normalize(obj, normalize_to=None):
    """Normalize the spectrogram or cumsum globally to a 0-1 range."""
    normalize = (obj - obj.min()) / (obj.max() - obj.min())
    if normalize_to:
        normalize=normalize*normalize_to         
    return (obj - obj.min()) / (obj.max() - obj.min())

def denormalize(tensor, original_min, original_max):
    return tensor * (original_max - original_min) + original_min

# def normalize_spectrogram_global(mel_spectrogram):
#     """Normalize the mel spectrogram globally to a 0-1 range."""
#     return (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())

def generate_spikes(audio_sample, mel_config, threshold, norm_inp=True, norm_cumsum=True):
    mel_spectrogram, sample_rate, _ = get_mel_spectrogram(audio_sample, mel_config)
    
    if norm_inp:
        original_min, original_max = mel_spectrogram.min(), mel_spectrogram.max()
        mel_spectrogram = normalize(mel_spectrogram)
    else:
        original_min, original_max = None, None  # No normalization applied
    
    delta_t = 1 / sample_rate
    csum = torch.cumsum(mel_spectrogram, dim=-1) * delta_t
    
    if norm_cumsum:
        csum_min, csum_max = csum.min(), csum.max()
        csum = normalize(csum)
    else:
        csum_min, csum_max = None, None  # No normalization applied to cumulative sum

    base_cum, pos_cum, neg_cum = step_forward_encoding(csum, threshold)
    spikes = pos_cum
    num_neurons = spikes.shape[0]
    num_spike_index = spikes.shape[1]
    
    return num_neurons, num_spike_index, spikes, (original_min, original_max), (csum_min, csum_max)

def inverse_generate_spikes(spikes, mel_config, sample_rate, threshold, 
                            norm_inp=True, norm_cumsum=True, 
                            orig_min_max=None, csum_min_max=None):
    original_min, original_max = orig_min_max if orig_min_max else (None, None)
    csum_min, csum_max = csum_min_max if csum_min_max else (None, None)
    
    delta_t = 1 / sample_rate
    reconstructed_csum = torch.cumsum(spikes * threshold, dim=-1) / delta_t
    
    # Undo cumulative sum to approximate the original mel spectrogram
    mel_spectrogram_approx = torch.diff(reconstructed_csum, dim=-1, prepend=torch.zeros(reconstructed_csum.shape[0], 1))

    # Undo normalization if it was applied to cumulative sum
    if norm_cumsum and csum_min is not None and csum_max is not None:
        mel_spectrogram_approx = denormalize(mel_spectrogram_approx, csum_min, csum_max)

    # Undo normalization if it was applied to the original mel spectrogram
    if norm_inp and original_min is not None and original_max is not None:
        mel_spectrogram_approx = denormalize(mel_spectrogram_approx, original_min, original_max)

    return mel_spectrogram_approx

def step_forward_encoding(batch, thr, neg=False):
    """
    Perform step-forward encoding on a batch of sequences.

    Args:
        batch (torch.Tensor): Input tensor of shape [batch_size, channels, features, time].
        thr (float): Threshold value for encoding spikes.
        neg (bool): If True, keep negative spikes as-is. Otherwise, convert them to positive magnitude.

    Returns:
        base_sig (torch.Tensor): Tensor of the reference signal (`base`) at each time step.
        pos_sig (torch.Tensor): Positive spike signals.
        neg_sig (torch.Tensor): Negative spike signals.
    """
    # Initialization
    L = batch.shape[-1]
    base = batch[..., 0]
    out = torch.zeros_like(batch)
    base_sig = torch.zeros_like(batch)

    # Main encoding loop
    for t in range(1, L):
        spikes = torch.where(batch[..., t] > base + thr, 1, 0) - torch.where(batch[..., t] < base - thr, 1, 0)
        base += spikes * thr
        out[..., t] = spikes
        base_sig[..., t] = base

    # Separate positive and negative spike signals
    pos_sig = torch.clamp(out, min=0)
    neg_sig = torch.where(out < 0, out if neg else -out, torch.zeros_like(out))
    
    return base_sig, pos_sig, neg_sig

def inverse_mel_to_waveform(mel_spectrogram, mel_config, sample_rate):
    """
    Approximate the original waveform from a mel spectrogram using InverseMelScale and GriffinLim.

    Args:
        mel_spectrogram (torch.Tensor): The mel spectrogram to convert, with shape (n_mels, time).
        mel_config: Configuration object with mel transform parameters (e.g., n_fft, n_mels, f_min, f_max).
        sample_rate (int): The sample rate of the original audio.

    Returns:
        torch.Tensor: The reconstructed waveform.
    """
    # Ensure mel_spectrogram is in the expected format (add batch dimension if needed)
    if mel_spectrogram.ndim == 2:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)  # Shape: (1, n_mels, time)

    # Step 1: Convert mel spectrogram to linear spectrogram
    inverse_mel_transform = T.InverseMelScale(
        n_stft=mel_config.n_fft // 2 + 1,
        n_mels=mel_config.n_mels,
        sample_rate=sample_rate,
        f_min=mel_config.f_min,
        f_max=mel_config.f_max,
    )
    linear_spectrogram = inverse_mel_transform(mel_spectrogram)

    # Step 2: Convert linear spectrogram to waveform using GriffinLim
    griffin_lim_transform = T.GriffinLim(n_fft=mel_config.n_fft, hop_length=mel_config.hop_length)
    waveform = griffin_lim_transform(linear_spectrogram)

    return waveform.squeeze()  # Remove batch dimension if present