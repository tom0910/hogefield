# zt_speech2spikes/zt_speech2spikes.py
from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch
import torchaudio

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, directory: str = "project/data/GSC", subset: str = None, download: bool = False):
        """
        A subset of the SPEECHCOMMANDS dataset.

        Parameters:
        - directory (str): Path to the dataset directory.
        - subset (str): Specify 'training', 'validation', or 'testing' subset.
        - download (bool): If True, downloads the dataset if not available.
        """
        # Ensure the directory exists or handle download manually
        if download and not os.path.exists(directory):
            raise ValueError(f"Download is not supported automatically in this class. Please ensure data exists in {directory}.")

        # Pass directory as the first positional argument to the parent class
        super().__init__(directory)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        # Adjust the walker based on the subset
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# class SubsetSC(SPEECHCOMMANDS):
#     # def __init__(self, subset: str = None):
#         super().__init__(directory, download=download)
#         # super().__init__("project/data/GSC", download=FALSE)

#         def load_list(filename):
#             filepath = os.path.join(self._path, filename)
#             with open(filepath) as fileobj:
#                 return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

#         if subset == "validation":
#             self._walker = load_list("validation_list.txt")
#         elif subset == "testing":
#             self._walker = load_list("testing_list.txt")
#         elif subset == "training":
#             excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
#             excludes = set(excludes)
#             self._walker = [w for w in self._walker if w not in excludes]

# Create training and testing split of the data. We do not use validation in this tutorial.
# train_set = SubsetSC("training")
# test_set = SubsetSC("testing")

# example to see your data:
# waveform, sample_rate, label, speaker_id, utterance_number = train_set[1]

if __name__ == "__main__":
    # Add any code that should only run when the script is executed directly
    pass

def print_my_name(text1,text2):
    print(f"{text1}{text2}zt_speec2spikes")

def get_unique_labels(dataset):
    """
    Get sorted list of unique labels from a dataset.

    Args:
        dataset: torch.utils.data.Dataset

    Returns:
        list: Sorted list of unique labels
    """
    return sorted(list(set(datapoint[2] for datapoint in dataset)))
# Usage
# train_set_labels = get_unique_labels(train_set)
# print(train_set_labels)

def standardize_batch(batch):
    mean = batch.mean(dim=(0, 3), keepdim=True)  # Compute mean along batch and time dimensions
    std = batch.std(dim=(0, 3), keepdim=True)  # Compute std along batch and time dimensions
    
    standardized_batch = (batch - mean) / std  # Standardize the batch
    return standardized_batch
# # USAGE:
# # Example batch data shape: [batch_size, channels, filterbank, time]
# batch_data = torch.randn(64, 1, 20, 201)  # Example batch data
# # Standardize the batch
# standardized_batch_data = standardize_batch(batch_data)


def calculate_num_frames(L, n_fft, hop_length, center=True):
    """
    Calculate the number of time frames in the spectrogram.

    Args:
    - L: Length of the input signal in samples.
    - n_fft: Window size.
    - hop_length: Hop length.
    - center: Whether padding is applied to the input signal (default: True).

    Returns:
    - Number of time frames.
    """
    if center:
        # Calculate the padded length
        padded_length = L + 2 * (n_fft // 2)
        num_frames = ((padded_length - n_fft) // hop_length) + 1
    else:
        num_frames = ((L - n_fft) // hop_length) + 1
    return num_frames
# USAGE:
# # Given parameters
# L = 16000  # Length of the input signal in samples
# n_fft = 400  # Window size
# hop_length = 80  # Hop length

# # Calculate the number of frames
# num_frames = calculate_num_frames(L, n_fft, hop_length, center=True)
# print("Number of time frames:", num_frames)


def gen_impulse_oscillator(analogue_in, num_bins=5, num_samples=10, test=None):
    # "analogue_in", input a normalized analogue input
    # "num_bins" is the possible frequency number for spikes
    # "num_samples" is the maximum number of delta tomes, devided per 2 gives the maximum frequency
    def map_to_bins(input_number, num_bins):
        if input_number < 0 or input_number > 1:
            print("input_number:",input_number)
            raise ValueError("Input number must be between 0 and 1")

        bin_size = 1 / num_bins
        bin_number = int(input_number // bin_size) + 1
        return min(bin_number, num_bins)

    def impulse_signal(num_samples, frequency):
        signal = np.zeros(num_samples)
        interval = int(num_samples / frequency)
        impulse_index = np.arange(0, num_samples, interval)
        signal[impulse_index] = 1
        return signal

    result=None
    if isinstance(analogue_in, float):
        result=impulse_signal(num_samples, map_to_bins(analogue_in, num_bins))
    elif isinstance(analogue_in, list):
        rec_signal=[]
        for i, analogue_input in enumerate(analogue_in):
            frq = map_to_bins(analogue_input, num_bins=num_bins)
            # if test==True: print(f'bin for input {analogue_input} is between {(frq-1)*(1/num_bins):.2f} and {frq*(1/num_bins):.2f}, so frequency class is: {frq}')
            result = impulse_signal(num_samples, frq)
            rec_signal.append(result)
        # Convert the list of results to a 2D PyTorch tensor
        result = torch.tensor(rec_signal)
        if test==True:
            fig, axs = plt.subplots(len(analogue_in), 1, figsize=(16, 16))
            plt.subplots_adjust(hspace=1)
            for i, analogue_input in enumerate(analogue_in):
                frq = map_to_bins(analogue_input, num_bins=num_bins)
                if test==True: print(f'bin for input {analogue_input} is between {(frq-1)*(1/num_bins):.2f} and {frq*(1/num_bins):.2f}, so frequency class is: {frq}')
                signal = impulse_signal(num_samples, frq)
                axs[i].stem(signal, linefmt='b-', basefmt='none')
                axs[i].set_title(f"Input: {analogue_input}, Bin: {frq}")
    elif isinstance(analogue_in, torch.Tensor):
        if analogue_in.dim() == 4:
            tensor3D=analogue_in.squeeze()
            rec_signal_spectograms = []
            for i_batch_idx, tensor2D in enumerate(tensor3D):
                rec_signal_rows = []
                for i_col_indx, tensor_list in enumerate(tensor2D):
                    rec_signal_row = []
                    for i, analogue_input in enumerate(tensor_list): 
                        frq = map_to_bins(analogue_input, num_bins=num_bins)
                        # if test==True: print(f'bin for input {analogue_input} is between {(frq-1)*(1/num_bins):.2f} and {frq*(1/num_bins):.2f}, so frequency class is: {frq}')
                        result = impulse_signal(num_samples, frq)
                        rec_signal_row.extend(result)
                    rec_signal_rows.append(rec_signal_row)  # Collect the results of each row
                # Convert the list of rows into a 2D PyTorch tensor
                rec_signal_tensor = torch.tensor(rec_signal_rows) 
                rec_signal_spectograms.append(rec_signal_tensor)
            # Stack the processed tensors back into a 3D tensor
            rec_signal_3D_tensor = torch.stack(rec_signal_spectograms)
            
            # Reshape the 3D tensor back to the original 4D shape
            rec_signal_4D_tensor = rec_signal_3D_tensor.unsqueeze(1)  # Add the channel dimension back
            
            result = rec_signal_4D_tensor 
            if test==True:
                print("rec_signal_4D has shape:",rec_signal_4D_tensor.shape)
                print(f'input shape for plot is:{rec_signal_4D_tensor[0].shape} and num_neurons is: {rec_signal_4D_tensor[0].squeeze(0).size(0)}')
                plot_pipline(rec_signal_4D_tensor[0],rec_signal_4D_tensor[0].squeeze(0).size(0))
        else:
            tensor2D=analogue_in.squeeze()
            rec_signal_rows = []
            for i_col_indx, tensor_list in enumerate(tensor2D):
                rec_signal_row = []
                for i, analogue_input in enumerate(tensor_list):
                    frq = map_to_bins(analogue_input, num_bins=num_bins)
                    # if test==True: print(f'bin for input {analogue_input} is between {(frq-1)*(1/num_bins):.2f} and {frq*(1/num_bins):.2f}, so frequency class is: {frq}')
                    result = impulse_signal(num_samples, frq)
                    rec_signal_row.extend(result)
                rec_signal_rows.append(rec_signal_row)  # Collect the results of each row
            # Convert the list of rows into a 2D PyTorch tensor
            rec_signal_tensor = torch.tensor(rec_signal_rows)   
            # print(rec_signal_tensor.shape)
            result=rec_signal_tensor
            if test==True:
                print(f'input shape for plot is: {result.unsqueeze(0).shape} and num_neurons is: {result.size(0)}')
                plot_pipline(result.unsqueeze(0),result.size(0))
        
            
    return result

def plot_pipline(tensor,num_neurons):
    # print("tensor that spikes:",tensor.shape)
    # print("neuron:",type(num_neurons))
    spike_data_sample = tensor[0,:,:].t()
    
    # print(spike_data_sample.shape)
    # print(spike_data_sample)
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)

    x, y = np.where(spike_data_sample)
    x *= 5
    # plt.xlim(0, num_times)  # Set x-axis limits from 0 to maximum x value + 1
    plt.ylim(0, num_neurons )  # Set y-axis limits from 0 to maximum y value + 1
    # print(len(y))
    # print(len(x))

    # # Get the indices of all elements (zero and non-zero)
    # indices = torch.nonzero(spike_data_sample)

    # # Extract x and y coordinates from the indices
    # x = indices[:, 1] * 5
    # y = indices[:, 0]
    
    # Plot raster plot
    # ax.scatter(*np.where(spike_data_sample), s=1.5, c="black")
    ax.scatter(x, y, s=1.5, c="black")
    print("HELLLO")

    plt.title("Input Spikes to SNN",fontsize=18)
    plt.xlabel("Time step (msec)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel("Neuron Number",fontsize=16)
    plt.yticks(fontsize=14)
    
    # plt.text(0.5, 0.5, 'Additional Info', fontsize=12, ha='center', va='center', transform=plt.gcf().transFigure, style='italic')
    
    plt.show()

def pipeline_a_data(waveform=None, train_set=None, label='bird', spectral_feature="Mel", hop_length=80, log_spectral_feature=False,threshold=0.247,cumsum = True, plot=False, sample_rate=None):
    """
    Pipeline data for processing or training.

    Parameters:
    - waveform (tensor): Waveform data to process.
    - train_set (list): List of data points for training.
    - label (str): Label for data retrieval if train_set is provided.
    - spectral_feature (str): Spectral feature to extract (options: 'Mel' or 'MFCC').
    - hop_length (int): Hop length for spectral feature extraction.
    - log_spectral_feature (bool): Whether to take the log of the spectral feature.
    - plot (bool): if true, it prints out the result
    - sample_rate (int): if given times are in seconds

    Returns:
    - processed waveform (tensor): Processed waveform data.
    Test Code:
        import zt_speech2spikes as s2s
        waveform=s2s.pipeline_a_data(your_waveform)
        s2s.check_tensor_values(waveform)
        print(waveform)
        waveform = s2s.pipeline_a_data(waveform=None, train_set=train_set, label='bird')
        print(waveform)
        waveform = s2s.pipeline_a_data(train_set=train_set, label='bird')
        print(waveform)
    """
    print("pipline a data func: I am from sppech2spike")
    if train_set is None:
        if waveform is None:
            raise ValueError("Either waveform or train_set must be provided.")
        
        # Main branch of the code
        # Process the waveform if train_set is not provided
        processed_waveform = waveform
    else:
        # Code to execute if train_set is provided
        # Retrieve waveform data based on the label
        result_data_points = get_data_by_label(dataset=train_set, target_label=label)
        processed_waveform, *_ = result_data_points[0]

    # Process the waveform
    processed_waveform = remove_bias_tesnor(processed_waveform)
    
    # Additional processing
    tensor = scale_tensor(processed_waveform)
    tensor = pad_sequence_tensor(tensor)
    if plot==True:
        plot_waveform(tensor.squeeze(0).numpy(), sample_rate=sample_rate, title_label="time domain plot")

    if spectral_feature == "Mel":
        Mel_transform = Mel_spec_transform(hop_length=hop_length)
        tensor = Mel_transform(tensor)
        if plot==True:
            #plot_mel_spectrogram(tensor, "Mel Transform")
            plot_mel_spectrogram2(tensor, "Mel Transform")
    elif spectral_feature == "MFCC":
        MFCC_transform = MFCC_transform(hop_length=hop_length, log_mels=log_spectral_feature)
        tensor = MFCC_transform(tensor)
        if plot==True:
            plot_mel_spectrogram(tensor, "MFCC transform")
    else:
        raise ValueError("Invalid param value. Supported values are 'Mel' and 'MFCC'")

    # tensor = normalize(tensor)
    _, pos_sig, pos_neg = step_forward_encoding(tensor, thr=threshold,neg=True)

    if cumsum:
        csum = torch.cumsum(tensor, dim=-1)
        _, pos_accum, neg_accum = step_forward_encoding(csum, thr=threshold,neg=True)

    processed_waveform = torch.cat((pos_sig, pos_neg, pos_accum, neg_accum), dim=1)
    if plot==True:
        # plot_mel_spectrogram(processed_waveform, "Spikes")
        plot_spike_heatmap(processed_waveform.squeeze(), cmap="binary", title="Spikes inputs neurons(time)")
        plot_pipline_(processed_waveform)
   
    return processed_waveform

def remove_bias_tesnor(tensor):
    bias = tensor.mean()
    tensor = tensor - bias
    return tensor

#Tested and working:
def remove_bias(tensors):
    # Ensure tensors is a list, even if a single tensor is passed
    if not isinstance(tensors, list):
        tensors = [tensors]

    rec_tensors = []
    for tensor in tensors:
        bias = tensor.mean()
        tensor = tensor - bias
        rec_tensors.append(tensor)

    # If a single tensor was passed, return it without wrapping in a list
    if len(rec_tensors) == 1:
        return rec_tensors[0]
    else:
        return rec_tensors

def plot_pipline_(tensor):
    spike_data_sample = tensor[0,:,:].t()

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)

    # Plot raster plot
    ax.scatter(*np.where(spike_data_sample), s=1.5, c="black")

    plt.title("Input to SNN")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()


def normalize(tensors):
    #std, mean, (data=data-mean)/std
    # print("to be normalized:",tensors.shape)
    mean_accumulator = 0.0
    std_accumulator = 0.0
    total_samples = 0
    batch_size = tensors.size(0)
    # print(batch_size)
    # print("batch_size:",batch_size)
    # print(waveform.shape)

    # batch_size is global that is not nice
    for tensor in tensors:
        total_samples += batch_size * tensor.numel()
        mean_accumulator += tensor.sum()
        std_accumulator += (tensor - tensor.mean(dim=2, keepdim=True)) ** 2

    mean = mean_accumulator / total_samples
    std = torch.sqrt(std_accumulator.sum() / total_samples)

    # print("Calculated mean:", mean.item()) #ez soknak tűnik
    # print("Calculated std:", std.item())  #ez soknak tűnik


    # sg=[(tensor - mean) / std for tensor in tensors]
    # print("SG:",len(sg))
    # print("SG1:",sg[0].shape)
    # sg2=torch.cat(sg,dim=0).unsqueeze(1)
    # print("SG2:",sg2.shape)

    rec_tensors = torch.cat([(tensor - mean) / std for tensor in tensors], dim=0).unsqueeze(1)
    return rec_tensors



# def scale_tensor(tensors):
#     # Ensure tensors is a list, even if a single tensor is passed
#     if not isinstance(tensors, list):
#         tensors = [tensors]

#     rec_tensors = []
#     desired_amplitude = 0.5
#     for tensor in tensors:
#         current_amplitude = tensor.max() - tensor.min()
#         scaling_factor = desired_amplitude / current_amplitude
#         tensor = tensor * scaling_factor
#         rec_tensors.append(tensor)

#     # If a single tensor was passed, return it without wrapping in a list
#     if len(rec_tensors) == 1:
#         return rec_tensors[0]
#     else:
#         return rec_tensors

def scale_tensor(tensor, desired_amplitude=0.5, desired_bias=0):
    abs_max = torch.max(torch.abs(tensor))
    scaling_factor = desired_amplitude / abs_max
    tensor = tensor * scaling_factor + desired_bias
    return tensor

def scale_tensor_list(tensor_list, desired_amplitude=0.5, desired_bias=0):
    if isinstance(tensor_list, list):
        return [scale_tensor(tensor, desired_amplitude, desired_bias) for tensor in tensor_list]
    else:
        return scale_tensor(tensor_list, desired_amplitude, desired_bias)
        
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)        

# Example usage:
# tensor_result = scale_tensor(single_tensor)
# list_result = scale_tensor(list_of_tensors)


# Define your function
def process_audio_tensors(tensors):
    processed_tensors = []
    for tensor in tensors:
        # Centering: Subtract the mean (bias)
        bias = np.mean(tensor)
        centered_tensor = tensor - bias
        
        # Scaling: Achieve a peak-to-peak amplitude of 0.5
        desired_amplitude = 0.5
        current_amplitude = centered_tensor.max() - centered_tensor.min()
        scaling_factor = desired_amplitude / current_amplitude
        scaled_tensor = centered_tensor * scaling_factor
        
        processed_tensors.append(scaled_tensor)
    return processed_tensors
    
# def step_forward_encoding(batch, thr): #batch looks like [32, 1, 20, 201]
#     L=batch.shape[-1] #Get the length (L) of the last dimension of the input batch tensor.
#     starts = batch[...,0]  #Extract the first element of each sequence in the batch and store it in the starts tensor.
#     out = torch.zeros_like(batch)

#     base = batch[..., 0]

#     for t in range(1, L):
#         spikes = torch.where(batch[..., t] > base + thr, 1, 0) + torch.where(batch[..., t] < base - thr, -1, 0)
#         base += spikes * thr
#         out[..., t] = spikes

#         pos_sig = torch.where(out > 0, out, 0)
#         neg_sig = torch.where(out < 0, -out, 0)
#     # Concatenate positive and negative signals along the feature dimension
#     out = torch.cat([pos_sig, neg_sig], dim=-2)

#     return out,pos_sig,neg_sig

import torch

def check_tensor_values(tensor):
    """
    Check the values within a tensor to determine if it contains only zeros and ones or if it contains
    zeros, ones, and minus ones.

    Parameters:
    - tensor (torch.Tensor): The input tensor to be checked.

    Returns:
    - str: A message indicating the nature of the values in the tensor.
    """
    if tensor.dim() not in {2, 3}:
        return "The function supports only 2D or 3D tensors."
    
    unique_values = torch.unique(tensor)
    
    if len(unique_values) == 2 and (0 in unique_values) and (1 in unique_values):
        return print("The tensor contains only zeros and ones.")
    elif len(unique_values) == 3 and (0 in unique_values) and (1 in unique_values) and (-1 in unique_values):
        return print("The tensor contains zeros, ones, and minus ones.")
    else:
        return print("The tensor does not meet the specified conditions.")

# # Example usage with tensors of different shapes
# tensor_ones_zeros = torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
# tensor_ones_zeros_and_minus_ones = torch.tensor([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
# tensor_other_values = torch.tensor([[0, 2, 1], [1, 0, 1], [0, 1, 0]])
# tensor_3d = torch.tensor([[[0, 1], [1, 0]], [[-1, 0], [0, -1]]])
# check_tensor_values(tensor_ones_zeros)
# check_tensor_values(tensor_ones_zeros_and_minus_ones)
# check_tensor_values(tensor_other_values)
# check_tensor_values(tensor_3d)


def step_forward_encoding(batch, thr, neg=None): #batch looks like [32, 1, 20, 201]
    L=batch.shape[-1] #Get the length (L) of the last dimension of the input batch tensor.
    starts = batch[...,0]  #Extract the first element of each sequence in the batch and store it in the starts tensor.
    out = torch.zeros_like(batch)

    base = batch[..., 0] # base(0)=batch(0) vagy B(1)=S(1) azaz a referencia inicializálva van bejövő jel első elemének értékével
    base_sig = torch.zeros_like(batch)
    # print("base:",base.shape)

    for t in range(1, L):
        spikes = torch.where(batch[..., t] > base + thr, 1, 0) + torch.where(batch[..., t] < base - thr, -1, 0) # ha S(t) nagyobb a referenciánál Th val, akkor 1, Ha kisebb Th-val -1, különben 0
        base = base + spikes * thr
        out[..., t] = spikes
        base_sig[..., t]=base

        pos_sig = torch.where(out > 0, out, 0)
        neg_sig = torch.where(out < 0, -out, 0)
        if neg==True:
            neg_sig = torch.where(out < 0, out, 0)
        else:
            neg_sig = torch.where(out < 0, -out, 0)    
            
    # Concatenate positive and negative signals along the feature dimension
    out = torch.cat([pos_sig, neg_sig], dim=-2)
    # print("base:",base.shape)
    return base_sig, pos_sig,neg_sig

def tensor_to_events(batch, threshold=1, device=None):
    """Converts a batch of continuous signals to binary spikes via delta
    modulation (https://en.wikipedia.org/wiki/Delta_modulation).

    Args:
        batch: PyTorch tensor of shape (..., timesteps)
        threshold: The difference between the residual and signal that
            will be considered an increase or decrease. Defaults to 1.
        device: A torch.Device used by PyTorch for the computation. Defaults to 
            None.

    Returns:
        A PyTorch int8 tensor of events of shape (..., timesteps).

    TODO:
        Add support for using multiple channels for polarity instead of signs
    """
    events = torch.zeros(batch.shape)
    levels = torch.round(batch[..., 0])
    if device:
        events = events.to(device)

    for t in range(batch.shape[-1]):
        events[..., t] = (batch[..., t] - levels > threshold).to(torch.int8) - (
            batch[..., t] - levels < -threshold
        ).to(torch.int8)
        levels += events[..., t] * threshold
    return events

def step_forward_encoding_with_accumulation(batch, thr):
    L = batch.shape[-1]
    starts = batch[..., 0]
    out = torch.zeros_like(batch)

    base = batch[..., 0]
    base_sig = torch.zeros_like(batch)
    pos_accum = torch.zeros_like(batch)
    neg_accum = torch.zeros_like(batch)

    for t in range(1, L):
        spikes = torch.where(batch[..., t] > base + thr, 1, 0) + torch.where(batch[..., t] < base - thr, -1, 0)
        base = base + spikes * thr
        out[..., t] = spikes
        base_sig[..., t] = base

        pos_spikes = torch.where(out > 0, out, 0)
        neg_spikes = torch.where(out < 0, -out, 0)

        pos_accum = pos_accum + pos_spikes  # Accumulate positive spikes over time
        neg_accum = neg_accum + neg_spikes  # Accumulate negative spikes over time

    # Concatenate positive and negative signals along the feature dimension
    out = torch.cat([pos_spikes, neg_spikes], dim=-2)

    return base_sig, pos_spikes, neg_spikes, pos_accum, neg_accum

import torch

def step_forward_encoding_single(batch, thr):
    L = batch.shape[-1]
    starts = batch[..., 0]
    out = torch.zeros_like(batch)

    base = batch[..., 0]
    base_sig = torch.zeros_like(batch)

    for t in range(1, L):
        spikes = torch.where(batch[..., t] > base + thr, 1, 0) + torch.where(batch[..., t] < base - thr, -1, 0)
        base = base + spikes * thr
        out[..., t] = spikes
        base_sig[..., t] = base

    pos_sig = torch.where(out > 0, out, 0)
    neg_sig = torch.where(out < 0, -out, 0)

    out = torch.cat([pos_sig, neg_sig], dim=-2)

    return base_sig, pos_sig, neg_sig

import matplotlib.pyplot as plt

def plot_spikes(spikes, s=1.0, c="black", title=None):
    """
    Plot spikes using plt.eventplot.

    Parameters:
    - spikes (torch.Tensor): Binary tensor indicating spikes (1 for spike, 0 for no spike).
    - s (float, optional): Marker size for the spikes.
    - c (str, optional): Color of the spikes.
    - title (str, optional): Title for the plot.

    Returns:
    - None
    """
    _, ax = plt.subplots()

    for i, spike_train in enumerate(spikes):
        spike_positions = torch.nonzero(spike_train).squeeze().numpy()
        ax.eventplot(spike_positions, colors=[c], lineoffsets=i, linelengths=s)

    ax.set_xlabel("Time")
    ax.set_ylabel("Neuron Index")
    ax.set_title(title)

    plt.show()

# Example usage:
# plot_spikes(d2, title="Positive Spikes")

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
import torch

def plot_spike_raster(input_tensor, title="Spike Raster", s=1.5, c="black"):
    """
    Plot a spike raster plot from a 3D input tensor.

    Parameters:
    - input_tensor (torch.Tensor): Input tensor of shape (batch_size, num_neurons, num_time_steps).
    - title (str, optional): Title for the plot.
    - s (float, optional): Marker size for the spikes.
    - c (str, optional): Color of the spikes.

    Returns:
    - None
    """
    # Ensure the input tensor is a PyTorch tensor
    # input_tensor = torch.tensor(input_tensor).clone().detach()
    input_tensor = input_tensor.clone().detach()
    # Reshape the tensor for raster plotting
    reshaped_tensor = input_tensor.squeeze(0).t()

    # Create a raster plot
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(reshaped_tensor, ax, s=s, c=c)

    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Neuron Number")

    plt.show()

# Example usage:
# plot_spike_raster(pos, title="Input Layer")





def sf(x,tr): #this does not seem working, base does not follow the signal
    out_pos = torch.zeros_like(x)
    out_neg = torch.zeros_like(x)
    base = torch.zeros_like(x)
    L = x.shape[-1]
    # B(1) = S(1) from publication
    base[...,0]=x[...,0]
    
    for t in range(1,L):
        # If the incoming signal intensity S(t1) exceeds the baseline B(t1−1) plus a threshold defined as Th, then a positive spike is encoded
        spikes_pos = torch.where(x[..., t] >  base[...,t-1] + tr, 1, 0)
        # if S(t1) <= B(t1−1) − Th, a negative spike is generated
        spikes_neg = torch.where(x[..., t] <= base[...,t-1] - tr, -1, 0)
        
        if torch.any(spikes_pos):  # Check if any spikes_pos is True
            #B(t1) is updated as B(t1) = B(t1−1) + Th;
            base[...,t]=base[...,t-1] + tr
    
        if torch.any(spikes_neg):  # Check if any spikes_neg is Tru
            #B(t1) is assigned as B(t1) = B(t1−1) − Th.
            base[...,t]=base[...,t-1] - tr
        # Check if neither positive nor negative spikes are detected
        # if not torch.any(spikes_pos) and not torch.any(spikes_neg):
        #     print(f"At time step {t}, neither positive nor negative spike is detected.")
            
        out_pos[..., t] = spikes_pos
        out_neg[..., t] = spikes_neg
    out = torch.cat([out_pos, out_neg], dim=0)

    # print("out:",out)
    return out,out_pos,out_neg

import torchaudio.transforms as T

def Mel_spec_transform(sample_rate=16000, n_mels=20, n_fft=512, f_min=20, f_max=4000, hop_length=80, test=False, mel_scale='htk'):
    """
    Returns a Mel spectrogram transformation with specified parameters.
    
    Parameters:
    - sample_rate (int, optional): The sample rate of the input waveform.
    - n_mels (int, optional): Number of mel filterbanks.
    - n_fft (int, optional): Number of Fourier bins.
    - f_min (float, optional): Minimum frequency.
    - f_max (float, optional): Maximum frequency.
    - hop_length (int, optional): The number of samples to step between frames.
    
    Returns:
    - torchaudio.transforms.MelSpectrogram: Mel spectrogram transformation.
    """    
    default_spec_kwargs = {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "f_min": f_min,
        "f_max": f_max,
        "hop_length": hop_length,
        "mel_scale": mel_scale,
    }
    transform = T.MelSpectrogram(**default_spec_kwargs)
    return transform

def MFCC_transform(sample_rate=16000, n_mfcc=20, log_mels=True, hop_length=80, n_mels=20, n_fft=512, f_min=20, f_max=4000):
    """
    Create an MFCC transformation with parameters.

    Args:
        sample_rate (int): Sample rate of the input waveform.
        n_mfcc (int): Number of mfc coefficients to retain
        log_mels (bool): Whether to use log-mel filterbanks.
        hop_length (int): Hop length for the STFT.
        n_mels (int): Number of Mel filterbanks.
        n_fft (int): Size of FFT.
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.

    Returns:
        torchaudio.transforms.MFCC: MFCC transformation with specified parameters.
    """
    default_spec_kwargs = {
        "sample_rate": sample_rate,
        "n_mfcc": n_mfcc,
        "log_mels": log_mels,
    }
    transform = torchaudio.transforms.MFCC(**default_spec_kwargs, melkwargs={"n_mels": n_mels, "n_fft": n_fft, "f_min": f_min, "f_max": f_max, "hop_length": hop_length})
    return transform

# def MFCC_transform(sample_rate=16000, n_mfcc=20, log_mels = "False", hop_length=80, n_mels=20,):
#     """
#     Create an MFCC transformation with parameters.

#     Args:
#         sample_rate (int): Sample rate of the input waveform.
#         n_mfcc (int): Number of mfc coefficients to retain
#         n_fft (int): Size of FFT.
#         min_frequency (float): Minimum frequency.
#         max_frequency (float): Maximum frequency.
#         hop_length (int): Hop length for the STFT.

#     Returns:
#         torchaudio.transforms.MFCC: MFCC transformation with specified parameters.
#     """
#     default_spec_kwargs = {
#         "sample_rate": sample_rate,
#         "n_mfcc": n_mfcc,
#         "log_mels": log_mels,
#     }
#     transform = torchaudio.transforms.MFCC(**default_spec_kwargs, melkwargs={"n_mels":n_mels, "n_fft":512, "f_min":20, "f_max":4000, "hop_length":hop_length})
#     return transform

import matplotlib.pyplot as plt

def plot_mel_spectrogram(mel_spectrogram, title):
    """
    Plot the Mel spectrogram using matplotlib.

    Parameters:
    - mel_spectrogram (torch.Tensor): Input Mel spectrogram tensor.
    - title (str): Title for the plot.

    Returns:
    - None
    """
    mel_spectrogram = mel_spectrogram.squeeze(0).numpy()  # Squeeze to remove the batch dimension
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mel Filter")
    plt.show()
    
def plot_spike_heatmap(spikes, cmap="binary", title=None):
    """
    Plot spikes as a heatmap.

    Parameters:
    - spikes (torch.Tensor): Binary tensor indicating spikes (1 for spike, 0 for no spike).
    - cmap (str, optional): Colormap for the heatmap. Default is "binary".
    - title (str, optional): Title for the plot.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(spikes, cmap=cmap, aspect="auto", interpolation="none",origin="lower")

    ax.set_xlabel("Time step")
    ax.set_ylabel("Neuron Number")
    ax.set_title(title)

    plt.show()    

def label_to_index(word,labels):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))
# def label_to_index(word, labels):
#     return torch.tensor(labels.index(word))  
    
def index_to_label(index, labels):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def size_check(tensors):

    for tensor in tensors:
        # print("mean:",tensor.mean())
        print("tensor max:",tensor.max())
        # print("tensor:min:",tensor.min())
        # print("tensor::::",tensor)
    
    tensors = [item.t() for item in tensors]
    element_lengths = [len(element) for element in tensors]
    if len(set(element_lengths)) > 1:
        print(element_lengths)
        raise ValueError("Not all elements in the batch have the same length!")
    else:
        print("batch data has same size:",element_lengths)

def pad_sequence_tensor(tensor, data_sample=16000):
    """
    Pad the input tensor (waveform shape ([1,sample_rate or length]))to the specified length along the second dimension using torch.nn.functional.pad.

    Parameters:
    - tensor (tensor): Input tensor to pad.
    - data_sample (int): Length to pad the tensor to along the second dimension.

    Returns:
    - padded tensor (tensor): Padded tensor.
    """
    # Get the current length of the tensor along the second dimension
    current_length = tensor.size(1)

    # If the tensor length is less than the specified length, pad it
    if current_length < data_sample:
        padding_needed = data_sample - current_length
        # Pad the tensor along the second dimension with zeros
        padded_tensor = torch.nn.functional.pad(tensor, (0, padding_needed), mode='constant', value=0.0)
    else:
        # If the tensor length is equal to or greater than the specified length, return the original tensor
        padded_tensor = tensor

    return padded_tensor

def zt_collate_fn(batch, cumsum, threshold, **kwargs):
    # print("batch_size type:",  type(batch))
    # print("batch_size length:",  len(batch))
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    # print("collate fn runs:")
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]
    
    tensors = remove_bias(tensors)
    # print_structure("bfr scale tensor shape:",tensors) # like list of Shape: torch.Size([1, 16000])
    tensors = scale_tensor(tensors) #bias should be included, but bias in these data set is neglectable
    # print_structure("after scale tensor shape:",tensors) # like list of Shape: torch.Size([1, 16000])
    tensors = pad_sequence(tensors)
    # size_check(tensors)
    Mel_transform=Mel_spec_transform()
    tensors=Mel_transform(tensors)
    # print("tensors sahpe after MEl",tensors.shape)
    # print(f"the mean is: {mean} and the std is: {std}")
    tensors=normalize(tensors)
    # print("tensor shape after normailized",tensors.shape)
    # print("length tensors after normalize",len(tensors))
    # print("type tensors after normalize",type(tensors))
    # this is done in Mel Transform:
    # tensors = torch.log(tensors)
    _,pos_sig,pos_neg= step_forward_encoding(tensors,threshold)
    # _,pos_sig,pos_neg= sf(tensors,threshold)
    if cumsum:
        csum = torch.cumsum(tensors, dim=-1)
        _,pos_accum,neg_accum= step_forward_encoding(csum,threshold)
        # _,pos_accum,neg_accum= sf(csum,threshold)
        # print("tensors_cum.shape:",tensors.shape)
    tensors = torch.cat((pos_sig,pos_neg,pos_accum,neg_accum), dim=2)
    # print("tensors shape after sf func.:",tensors.shape)

    # Group the list of tensors into a batched tensor
    # tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    # print("printing targest:",targets)
    return tensors, targets

# you can do this for using collate_fn in your code:
# import torch
# import zt_speech2spikes

# def custom_collate_fn(batch):
#     cumsum = True
#     threshold = 0.47
    
#     batch_size = len(batch)
#     tensors, targets = [], []

#     for waveform, _, label, *_ in batch:
#         # print(label)
#         tensors += [waveform]
#         targets += [zt_speech2spikes.label_to_index(label,target_labels)]

#     tensors = zt_speech2spikes.remove_bias(tensors)
#     tensors = zt_speech2spikes.scale_tensor(tensors)
#     tensors = zt_speech2spikes.pad_sequence(tensors)
    
#     Mel_transform = zt_speech2spikes.Mel_spec_transform()
#     tensors = Mel_transform(tensors)
    
#     tensors = zt_speech2spikes.normalize(tensors)
#     _, pos_sig, pos_neg = zt_speech2spikes.step_forward_encoding(tensors, threshold)

#     if cumsum:
#         csum = torch.cumsum(tensors, dim=-1)
#         _, pos_accum, neg_accum = zt_speech2spikes.step_forward_encoding(csum, threshold)

#     tensors = torch.cat((pos_sig, pos_neg, pos_accum, neg_accum), dim=2)
#     targets = torch.stack(targets)

#     return tensors, targets

# # Assuming you have already imported the necessary modules and defined your train_set
# train_loader = torch.utils.data.DataLoader(
#     train_set,
#     batch_size=32,
#     shuffle=True,
#     collate_fn=custom_collate_fn,
# )

# batch = next(iter(train_loader))

def get_data_by_label(dataset, target_label, num_instances=1):
    found_instances = 0
    results = []

    for data_point in dataset:        
        _, _, label, _, _ = data_point

        # Check if the label matches the target_label
        if label == target_label:
            results.append(data_point)
            found_instances += 1

            # Break the loop if the desired number of instances is reached
            if found_instances >= num_instances:
                break

    # Return the list of found instances
    return results

# import matplotlib.pyplot as plt
# import torch
# import numpy as np

# def plot_waveform(waveform, sample_rate, title_label=None):
#     # print(type(waveform))
#     # print(waveform.shape)
#     # waveform_np = waveform.squeeze(0)
#     # print(waveform_np.shape)
#     # print(waveform_np.numpy())
#     """
#     Plot the waveform using matplotlib.

#     Parameters:
#     - waveform (torch.Tensor): Input waveform tensor.
#     - sample_rate (int): Sampling rate of the waveform.
#     - title_label (str, optional): Label to include in the title.

#     Returns:
#     - None
#     """
#     duration = len(waveform) / sample_rate
#     time_axis = np.linspace(0, duration, len(waveform))
#     print(type(waveform))
#     print(len(waveform))
#     waveform_np = waveform.squeeze(0).numpy()
#     print(waveform_np)

#     plt.figure(figsize=(10, 4))
#     plt.plot(time_axis, waveform_np  # Assuming waveform is a PyTorch tensor
#     plt.title("Waveform" + (f" - {title_label}" if title_label else ""))
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.show()

# # Example usage:
# # Assuming 'your_waveform' is a torch.Tensor, 'your_sample_rate' is an integer,
# # and 'your_label' is a string
# # plot_waveform(your_waveform, sample_rate=your_sample_rate, title_label=your_label)

import numpy as np
import matplotlib.pyplot as plt

def plot_waveform(waveform, sample_rate=None, title_label=None):
    if sample_rate is not None:
        duration = len(waveform) / sample_rate
    else:
        duration = len(waveform)
    time_axis = np.linspace(0, duration, len(waveform))

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, waveform)  # Remove .numpy() here
    plt.title("Waveform" + (f" - {title_label}" if title_label else ""))
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def plot_mel_spectrogram2(mel_spectrogram, f_min, f_max,  title, cmap='plasma', windows_width=13):
    """
    Plot the Mel spectrogram using matplotlib.

    Parameters:
    - mel_spectrogram (torch.Tensor): Input Mel spectrogram tensor.
    - title (str): Title for the plot.
    - cmap: 'viridis', 'magma', 'plasma', 'cividis'

    Returns:
    - None
    """
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)
    # Squeeze to remove the batch dimension
    mel_spectrogram = mel_spectrogram.squeeze(0).numpy()  

    # Get the number of Mel filters and time steps
    num_mels, num_timesteps = mel_spectrogram.shape
    
    # Set the width of the figure
    width = windows_width
    height = 2.5* width * (num_mels / num_timesteps)

    # Create the plot with the specified width and height
    plt.figure(figsize=(width, height))
    plt.imshow(mel_spectrogram, aspect="auto", origin="lower", cmap=cmap)

    # Add colorbar
    plt.colorbar(label='Magnitude')

    # Set title and labels
    plt.title(title, fontsize=18)
    plt.xlabel("Time", fontsize=16)
    # plt.ylabel("Mel Filter", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # # Annotate y-axis with frequencies corresponding to Mel filterbanks
    # mel_indices = np.arange(num_mels)
    # frequencies = mel_to_hz(mel_indices)
    # plt.yticks(mel_indices, [f'{frequency:.0f} Hz' for frequency in frequencies])
    
    # Annotate y-axis with frequencies corresponding to Mel filterbanks
    mel_min = f_min  # Minimum Mel frequency
    mel_max = hz_to_mel(f_max)  # Maximum Mel frequency corresponding to 8000 Hz
    mel_values = np.linspace(mel_min, mel_max, num_mels)  # Generate Mel frequencies
    frequencies = mel_to_hz(mel_values)  # Convert Mel frequencies to Hz
    plt.yticks(np.arange(num_mels), [f'{frequency:.0f} Hz' for frequency in frequencies])

    # Remove x-axis
    plt.gca().axes.get_xaxis().set_visible(False)    

    plt.show()