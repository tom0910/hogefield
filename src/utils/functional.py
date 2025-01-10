import torch as torch
from torchaudio.transforms import Spectrogram
from core.CustomMelScale import CustomMelScale
import torchaudio.transforms as T
import utils.preprocess_collate as PC

def pad_waveform_to_length(waveform, to=16000):

    if waveform.size(0) < to:
        padded_waveform = torch.cat([waveform, torch.zeros(to - waveform.size(0))], dim=0)
    else:
        padded_waveform = waveform[:to]
    return padded_waveform

def get_mel_spectrogram(audio_sample, mel_config):
    """
    Compute the mel spectrogram using the filter type specified in mel_config.
    """
    waveform, sample_rate = audio_sample.load_waveform()

    waveform = pad_waveform_to_length(waveform=waveform)

    #debug:
    # mel_spectrogram = mel_config.debug_transform(waveform)        
    
    # # Apply Spectrogram transformation first, then use CustomMelScale for filtering
    # spectrogram_transform = Spectrogram(n_fft=mel_config.n_fft, hop_length=mel_config.hop_length, power=mel_config.power)
    # spectrogram = spectrogram_transform(waveform)    

    # # Initialize CustomMelScale with the filter type and configuration
    # custom_mel_scale = CustomMelScale(
    #     n_mels=mel_config.n_mels,
    #     sample_rate=mel_config.sample_rate,
    #     f_min=mel_config.f_min,
    #     f_max=mel_config.f_max,
    #     n_stft=mel_config.n_fft // 2 + 1,
    #     filter_type=mel_config.filter_type
    # )

    # # # Apply the CustomMelScale transformation to the spectrogram
    # mel_spectrogram = custom_mel_scale(spectrogram)
    
    mel_spectrogram, sample_rate, custom_mel_scale = PC.gen_melspecrogram_common(
        waveform, mel_config.n_mels, sample_rate, mel_config.f_min, mel_config.f_max, 
        mel_config.filter_type, mel_config.n_fft, mel_config.hop_length, mel_config.power, center=True
        )
    
    return mel_spectrogram, sample_rate, custom_mel_scale

def get_mel_spectrogram2(audio_sample, mel_config):
    #calls gen_melspecrogram_common2() kontra gen_melspecrogram_common()
    """
    Compute the mel spectrogram using the filter type specified in mel_config.
    """
    waveform, sample_rate = audio_sample.load_waveform()

    waveform = pad_waveform_to_length(waveform=waveform)
    
    mel_spectrogram, sample_rate, custom_mel_scale = PC.gen_melspecrogram_common2(
        waveform, mel_config.n_mels, sample_rate, mel_config.f_min, mel_config.f_max, 
        mel_config.filter_type, mel_config.n_fft, mel_config.hop_length, mel_config.power, center=True
        )
    
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

# def normalize(obj, normalize_to=None):
#     """Normalize the spectrogram or cumsum globally to a 0-1 range."""
#     normalize = (obj - obj.min()) / (obj.max() - obj.min())
#     if normalize_to:
#         normalize=normalize*normalize_to         
#     return (obj - obj.min()) / (obj.max() - obj.min())

# Original normalization function
def normalize(obj, normalize_to=None):
    """Normalize the spectrogram or cumsum globally to a 0-1 range."""
    normalized = (obj - obj.min()) / (obj.max() - obj.min())
    if normalize_to:
        normalized = normalized * normalize_to
    return normalized

def inverse_normalize(normalized_obj, original_min, original_max, normalize_to=None):
    """Invert the normalization to restore the original range."""
    if normalize_to:
        normalized_obj = normalized_obj / normalize_to
    return normalized_obj * (original_max - original_min) + original_min

def denormalize(tensor, original_min, original_max):
    return tensor * (original_max - original_min) + original_min

# def normalize_spectrogram_global(mel_spectrogram):
#     """Normalize the mel spectrogram globally to a 0-1 range."""
#     return (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())

# original working for preprocess examination of main_demonstrate_s2s.py 
def generate_spikes(audio_sample, mel_config, threshold, norm_inp=False, norm_cumsum=False):
    
    mel_spectrogram, sample_rate, _ = get_mel_spectrogram(audio_sample, mel_config)
    #normalize melspectrogram output
    # mel_spectrogram = normalize(mel_spectrogram) # do not normalize twice!!!
    
    ##  SET TO FALSE IN SIGNITURE ALWAYS:
    if norm_inp:
        original_min, original_max = mel_spectrogram.min(), mel_spectrogram.max()
        print("The original max {original_max} and min {original_min} value of mel-spectrogram") 
        mel_spectrogram = normalize(mel_spectrogram)
    else:
        original_min, original_max = None, None  # No normalization applied
 
    delta_t = 1 / sample_rate
    csum = torch.cumsum(mel_spectrogram, dim=-1) * delta_t
    
    ##  SET TO FALSE IN SIGNITURE ALWAYS:
    if norm_cumsum:
        csum_min, csum_max = csum.min(), csum.max()
        csum = normalize(csum)
    else:
        csum_min, csum_max = None, None  # No normalization applied to cumulative sum

    base_cum, pos_cum, neg_cum = step_forward_encoding(batch=csum, thr=threshold, neg=False)
    spikes = pos_cum
    num_neurons = spikes.shape[0]
    num_spike_index = spikes.shape[1]
    print(f"spikes type from generate_spikes() func:{type(spikes)} and shape:{spikes.shape}")
    return num_neurons, num_spike_index, spikes, (original_min, original_max), (csum_min, csum_max)


def compute_cumulative_sum(signal, delta_t):
    """
    Compute the cumulative sum of a signal and scale by delta_t.
    Args:
        signal (torch.Tensor): The input signal.
        delta_t (float): Scaling factor.
    Returns:
        torch.Tensor: Cumulative sum of the signal.
    """
    return torch.cumsum(signal, dim=-1) * delta_t




def reconstruct_from_cumulative_sum(cusm, delta_t):
    """
    Reconstruct the original signal from its cumulative sum.
    Args:
        cusm (torch.Tensor): The cumulative sum of the signal.
        delta_t (float): Scaling factor used during cumulative sum computation.
    Returns:
        torch.Tensor: Reconstructed original signal.
    """
    return torch.diff(cusm, dim=-1, prepend=torch.zeros_like(cusm[..., :1])) / delta_t

from scipy.ndimage import gaussian_filter1d
def smooth_cumulative_sum(csum, sigma):
    """
    Smooth a cumulative sum using Gaussian smoothing.

    Args:
        csum (torch.Tensor): The input cumulative sum tensor.
        sigma (float): The standard deviation for the Gaussian kernel.

    Returns:
        torch.Tensor: The smoothed cumulative sum.
    """
    # Ensure input is a 1D or 2D tensor
    if csum.ndim == 1:
        csum = csum.unsqueeze(0)  # Add a batch dimension if necessary

    # Apply Gaussian smoothing along the last dimension
    smoothed = torch.tensor(gaussian_filter1d(csum.numpy(), sigma=sigma, axis=-1))

    return smoothed

def differentiate_smoothed_csum(smoothed_csum, delta_t):
    """
    Differentiate the smoothed cumulative sum.

    Args:
        smoothed_csum (torch.Tensor): The smoothed cumulative sum.
        delta_t (float): The time step between samples.

    Returns:
        torch.Tensor: The derivative of the smoothed cumulative sum.
    """
    # Calculate the differences along the last dimension
    delta_y = torch.diff(smoothed_csum, dim=-1, prepend=torch.zeros_like(smoothed_csum[..., :1]))
    
    # Divide by delta_t to compute the derivative
    derivative = delta_y / delta_t
    
    return derivative


# original working for preprocess examination of main_demonstrate_s2s.py 
def generate_spikes2(audio_sample, mel_config, threshold, norm_inp=False, norm_cumsum=False):
    """
      Uses standard cumulative sum and inverse_cumulative sum functions 
      for best performance: normalizes FFT frames (in get_mel_spectrogram2()) and mel-spectrogram (norm_inp=True) when called
      no cumulative sum is normalized: norm_cumsum=False
    """
    
    mel_spectrogram, sample_rate, _ = get_mel_spectrogram2(audio_sample, mel_config)
    #normalize melspectrogram output
    # mel_spectrogram = normalize(mel_spectrogram)
    # mel_spectrogram = normalize_spectrogram_global(mel_spectrogram)
    
    ##  SET TO FALSE IN SIGNITURE ALWAYS:
    if norm_inp:
        # just only return these values
        original_min, original_max = mel_spectrogram.min(), mel_spectrogram.max()
        # normalization happens here
        mel_spectrogram = normalize(mel_spectrogram)
    else:
        original_min, original_max = None, None  # No normalization applied
 
    print("before csum the signal to be spiked",mel_spectrogram.shape)
    delta_t = 1 / sample_rate
    print("delta_t:",delta_t)
    csum = compute_cumulative_sum(signal=mel_spectrogram, delta_t=delta_t)
    # csum = torch.cumsum(mel_spectrogram, dim=-1) * delta_t
    # print("csum shape , the signal to be spiked",csum.shape)
    
    ##  SET TO FALSE IN SIGNITURE ALWAYS:
    if norm_cumsum:
        csum_min, csum_max = csum.min(), csum.max()
        csum = normalize(csum)
    else:
        csum_min, csum_max = None, None  # No normalization applied to cumulative sum

    base_cum, pos_cum, neg_cum = step_forward_encoding(batch=csum, thr=threshold, neg=False)
    spikes = pos_cum
    num_neurons = spikes.shape[0]
    num_spike_index = spikes.shape[1]
    print(f"spikes type from generate_spikes() func:{type(spikes)} and shape:{spikes.shape}")
    return num_neurons, num_spike_index, spikes, (original_min, original_max), (csum_min, csum_max)


def generate_spikes_of_audio(audio_sample, mel_config, threshold, norm_inp=False, norm_cumsum=False):
    
    # mel_spectrogram, sample_rate, _ = get_mel_spectrogram2(audio_sample, mel_config)
    waveform, sample_rate = audio_sample.load_waveform()
    #normalize melspectrogram output
    # mel_spectrogram = normalize(mel_spectrogram)
    # mel_spectrogram = normalize_spectrogram_global(mel_spectrogram)
    
    ##  SET TO FALSE IN SIGNITURE ALWAYS:
    if norm_inp:
        # just only return these values
        original_min, original_max = waveform.min(), waveform.max()
        # normalization happens here
        waveform = normalize(waveform)
    else:
        original_min, original_max = None, None  # No normalization applied
 
    
    delta_t = 1 / sample_rate
    csum = torch.cumsum(waveform, dim=-1) * delta_t
    
    ##  SET TO FALSE IN SIGNITURE ALWAYS:
    if norm_cumsum:
        csum_min, csum_max = csum.min(), csum.max()
        csum = normalize(csum)
    else:
        csum_min, csum_max = None, None  # No normalization applied to cumulative sum

    base_cum, pos_cum, neg_cum = step_forward_encoding(batch=csum, thr=threshold, neg=False)
    spikes = pos_cum
    spikes = spikes[0:1000]
    print("spikes.shape:",spikes.shape)
    # num_neurons = spikes.shape[0]
    num_neurons = 1
    # num_spike_index = spikes.shape[1]
    spikes = spikes.unsqueeze(0)
    
    num_spike_index = spikes.shape[1]
    print(f"spikes type from generate_spikes() func:{type(spikes)} and shape:{spikes.shape}")
    return num_neurons, num_spike_index, spikes, (original_min, original_max), (csum_min, csum_max)


import utils.preprocess_collate as PC
# developped under the test for preprocess examination to replace  generate_spikes in main_demonstrate_s2s.py
def generate_spikes_from_audio(audio_sample, mel_config, threshold, norm_inp=None, norm_cumsum=None):
    """
    Generates spikes from a single audio_sample using preprocess_collate and returns results
    in the format of generate_spikes.

    Args:
        audio_sample: Single audio sample to process.
        mel_config: Configuration object with spectrogram parameters.
        threshold: Threshold for spike encoding.

    Returns:
        num_neurons: Number of neurons in the spike data.
        num_spike_index: Spike index information.
        spikes: Encoded spike data.
        (original_min, original_max): Min and max of mel-spectrogram before normalization.
        (csum_min, csum_max): Min and max of cumulative sum.
    """
    
    # Extract the waveform from the audio sample
    waveform, sample_rate = audio_sample.load_waveform()

    # Create a fake batch: Wrap the waveform in a list
    tensors = [waveform.unsqueeze(0)]  # Shape: [1, samples]
    targets = [0]  # Dummy target for compatibility with preprocess_collate

    
    # Extract parameters from mel_config
    n_fft = mel_config.n_fft
    hop_length = mel_config.hop_length
    n_mels = mel_config.n_mels
    sample_rate = mel_config.sample_rate
    f_min = mel_config.f_min
    f_max = mel_config.f_max
    filter_type = mel_config.filter_type

    # Call preprocess_collate without changing it
    spikes, _, num_neurons, base_cums = PC.preprocess_collate(
        tensors, targets, n_fft, hop_length, n_mels, sample_rate, f_min, f_max, threshold, filter_type
    )
    
    # Remove batch and channel dimensions
    spikes = spikes.squeeze(0).squeeze(0)  # Shape becomes [22, 801]    
    
    print(f"spikes type from generate_spikes_from_audio() func:{type(spikes)} and shape:{spikes.shape}")
    # Calculate additional values
    original_min, original_max = spikes.min().item(), spikes.max().item()
    csum_min, csum_max = base_cums.min().item(), base_cums.max().item()

    # Return in the format of generate_spikes
    num_spike_index = spikes.shape[1]  # Spike index dimension
    return num_neurons, num_spike_index, spikes, (original_min, original_max), (csum_min, csum_max)



def inverse_generate_spikes(spikes, mel_config, sample_rate, threshold, 
                            norm_inp=True, norm_cumsum=True, 
                            orig_min_max=None, csum_min_max=None):
    original_min, original_max = orig_min_max if orig_min_max else (None, None)
    csum_min, csum_max = csum_min_max if csum_min_max else (None, None)
    
    delta_t = 1 / sample_rate
    # reconstructed_csum = 10
    reconstructed_csum = torch.cumsum(spikes * threshold, dim=-1) / delta_t
    
    # Undo cumulative sum to approximate the original mel spectrogram
    mel_spectrogram_approx = torch.diff(reconstructed_csum, dim=-1, prepend=torch.zeros(reconstructed_csum.shape[0], 1))

    # Undo normalization if it was applied to cumulative sum
    if norm_cumsum and csum_min is not None and csum_max is not None:
        mel_spectrogram_approx = inverse_normalize(mel_spectrogram_approx, csum_min, csum_max)

    # Undo normalization if it was applied to the original mel spectrogram
    if norm_inp and original_min is not None and original_max is not None:
        mel_spectrogram_approx = inverse_normalize(mel_spectrogram_approx, original_min, original_max)

    return mel_spectrogram_approx

def inverse_step_forward(pos_sig, thr):
    base = torch.zeros_like(pos_sig[..., 0])
    csum = torch.zeros_like(pos_sig)
    for t in range(1, pos_sig.shape[-1]):
        base += pos_sig[..., t] * thr
        csum[..., t] = base
    return csum

def inverse_generate_spikes2(spikes, mel_config, sample_rate, threshold, 
                            norm_inp=True, norm_cumsum=True, 
                            orig_min_max=None, csum_min_max=None, norm_out=None):
    original_min, original_max = orig_min_max if orig_min_max else (None, None)
    csum_min, csum_max = csum_min_max if csum_min_max else (None, None)
    
    if orig_min_max:
        print("original min:", original_min, "original max:", original_max)
    
    delta_t = 1 / sample_rate
    # reconstructed_csum has been changed compared what it was in inverse_generate_spikes2()
    reconstructed_csum= inverse_step_forward(pos_sig=spikes,thr=threshold)
    # in old code
    # reconstructed_csum = torch.cumsum(spikes * threshold, dim=-1) / delta_t
    
    smoothed_cusm= smooth_cumulative_sum(csum=reconstructed_csum,sigma=5)
    mel_spectrogram_approx=differentiate_smoothed_csum(smoothed_csum=smoothed_cusm,delta_t=delta_t)
    # gain back mel-spectrogram by inverse cumulative sum
    # mel_spectrogram_approx = reconstruct_from_cumulative_sum(cusm=reconstructed_csum, delta_t=delta_t)
    
    # Undo cumulative sum to approximate the original mel spectrogram
    # mel_spectrogram_approx = torch.diff(reconstructed_csum, dim=-1, prepend=torch.zeros(reconstructed_csum.shape[0], 1))

    # Undo normalization if it was applied to cumulative sum
    if norm_cumsum and csum_min is not None and csum_max is not None:
        print("denormalize csum is done")
        mel_spectrogram_approx = denormalize(mel_spectrogram_approx, csum_min, csum_max)

    # Undo normalization if it was applied to the original mel spectrogram
    if norm_inp and original_min is not None and original_max is not None:
        print("INVERSE NORMALIZATION OF MEL_SPECTROGRAM is done!")
        print(f"basec on max:{original_max} and min: {original_min} values")
        mel_spectrogram_approx = denormalize(mel_spectrogram_approx, original_min, original_max)
        
    if norm_out:
        mel_spectrogram_approx= normalize(mel_spectrogram_approx)

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
        # print(f"is string?{type(thr)} and {type(base)}")
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

def inverse_mel_to_waveform2(mel_spectrogram, mel_config, sample_rate, filter_type="standard"):
    """
    Approximate the original waveform from a mel spectrogram using InverseMelScale and Griffin-Lim.

    Args:
        mel_spectrogram (torch.Tensor): The mel spectrogram to convert, shape (n_mels, time).
        mel_config: Configuration object with mel transform parameters (e.g., n_fft, n_mels, f_min, f_max).
        sample_rate (int): Sample rate of the original audio.
        filter_type (str): Filter type used during forward mel spectrogram creation.
    
    Returns:
        torch.Tensor: The reconstructed waveform.
    """
    if mel_spectrogram.ndim == 2:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)  # Shape: (1, n_mels, time)

    # Step 1: Convert mel spectrogram to linear spectrogram
    if filter_type == "standard":
        inverse_mel_transform = T.InverseMelScale(
            n_stft=mel_config.n_fft // 2 + 1,
            n_mels=mel_config.n_mels,
            sample_rate=sample_rate,
            f_min=mel_config.f_min,
            f_max=mel_config.f_max,
            norm=None
        )
        linear_spectrogram = inverse_mel_transform(mel_spectrogram)
    # elif filter_type == "custom":
    #     # Implement the inverse of the custom mel filter bank
    #     linear_spectrogram = custom_inverse_mel_filter(
    #         mel_spectrogram,
    #         mel_config,
    #         sample_rate
    #     )
    else:
        raise ValueError(f"Unsupported filter_type: {filter_type}")

    # Step 2: Convert linear spectrogram to waveform using Griffin-Lim
    griffin_lim_transform = T.GriffinLim(n_fft=mel_config.n_fft, hop_length=mel_config.hop_length)
    waveform = griffin_lim_transform(linear_spectrogram)

    return waveform.squeeze()  # Remove batch dimension if present



from datetime import datetime
def on_click_save_params(params, ID):
    """
    Save parameters to a file.

    Parameters:
    - params (dict): The parameters to save.
    - file_path (str): The file path where parameters will be saved.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path=f"id_{ID}_{timestamp}_params.txt"
    try:
        with open(file_path, "w") as file:
            for key, value in params.items():
                file.write(f"{key}: {value}\n")
        print(f"Parameters saved to {file_path}")
    except Exception as e:
        print(f"Error saving parameters: {e}")