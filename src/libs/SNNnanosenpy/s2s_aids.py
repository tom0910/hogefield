# s2s_aids/zt_s2s_aids

#!!!! javítások:
# duplikáltan a speech2spikes-ban van függyvény ne legyen ebben a könyvtában
from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch
import torchaudio
import zt_speech2spikes as s2s
import matplotlib.pyplot as plt
import numpy as np


def gen_impulse_oscillator(analogue_in, num_bins=5, num_samples=10, test=None):
    # "analogue_in", input a normalized analogue input
    # "num_bins" is the possible frequency number for spikes
    # "num_samples" is the maximum number of delta tomes, devided per 2 gives the maximum frequency
    def map_to_bins(input_number, num_bins):
        if input_number < 0 or input_number > 1:
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
            plot_pipline(result.unsqueeze(0),result.size(0))
        
            
    return result

# Might cause issue. Because there is an other function with the same name
def plot_pipline(tensor,num_neurons,windows_width=10.5):
    # print("tensor that spikes:",tensor.shape)
    # print("neuron:",type(num_neurons))
    spike_data_sample = tensor[0,:,:].t()
    
    # print(spike_data_sample.shape)
    # print(spike_data_sample)

    height = 0.3 * windows_width
    
    fig = plt.figure(facecolor="w", figsize=(windows_width, height))
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

    plt.title("Input Spikes to SNN",fontsize=18)
    plt.xlabel("Time step (msec)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel("Neuron Number",fontsize=16)
    plt.yticks(fontsize=14)
    
    # plt.text(0.5, 0.5, 'Additional Info', fontsize=12, ha='center', va='center', transform=plt.gcf().transFigure, style='italic')
    
    plt.show()

def remove_bias_tesnor(tensor):
    bias = tensor.mean()
    tensor = tensor - bias
    return tensor

# def my_MFCC_transform(sample_rate=16000, n_mfcc=20, log_mels=False, hop_length=80, n_mels=20, n_fft=512, f_min=20, f_max=4000):
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
#     # transform = torchaudio.transforms.MFCC(**default_spec_kwargs, melkwargs={"n_mels":n_mels, "n_fft":512, "f_min":20, "f_max":4000, "hop_length":hop_length})
#     transform = torchaudio.transforms.MFCC(**default_spec_kwargs, melkwargs={"n_mels":n_mels, "n_fft":n_fft, "f_min":f_min, "f_max":f_max, "hop_length":hop_length})
#     return transform

import matplotlib.pyplot as plt

def plot_waveform(waveform, sample_rate=None, title_label=None, windows_width=10):
    if sample_rate is not None:
        duration = len(waveform) / sample_rate
    else:
        duration = len(waveform)
    time_axis = np.linspace(0, duration, len(waveform))

    height = 0.3* windows_width
    
    plt.figure(figsize=(windows_width, height))
    plt.plot(time_axis, waveform)  # Remove .numpy() here
    plt.title("Waveform" + (f" - {title_label}" if title_label else ""), fontsize=18)
    plt.xlabel("Time step (msec)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel("Neuron Number",fontsize=16)
    plt.yticks(fontsize=14)

    # Remove x-axis
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.show()

from snntorch import spikegen

def pipeline_a_data(waveform=None, train_set=None, label='bird', spectral_feature="Mel", hop_length=80, log_spectral_feature=False,threshold=0.247,cumsum = True, plot=False, sample_rate=None, n_mels=20, f_min=20, f_max=4000, spike_encoding="step_forward", SpikingGenOffset=1):
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
    print("pipeline_a_data function from s2s_aids")
    print("from GSC dataset the word is:", label)
    if spike_encoding=="step_forward": print("step forward applied")
    print("spectral feature:", spectral_feature)
    if spectral_feature=="MFCC":
        print(f'set log_mel that is now: {log_spectral_feature}  whether to use log-mel spectrograms instead of db-scaled. (Default: False)')
    print("threshold:", threshold) if spike_encoding=="step_forward" else print("there is no threshold set for speec2spikes encodeing")
    print("fmin:",f_min,"|fmax:",f_max,"|n_mels:",n_mels,"|speech2spikes pipeline is used:",spike_encoding=="step_forward","|time_step of spectogramm:",hop_length*1000/sample_rate,"msec","\n|hop_length:",hop_length)

    not_sf = spike_encoding=="step_forward"
    is_sf  = spike_encoding=="step_forward"
    variables = {
        'from GSC dataset the word is: ': {'value': label, 'exclude': False},
        'pipeline :' : {'value': 'step forward applied', 'exclude': not_sf},  # Excluded variable if
        'spectral feature: ': {'value': spectral_feature, 'exclude': False},  
        'log_mel: ': {'value':  log_spectral_feature, 'exclude': spectral_feature=="MFCC"},
        'threshold': {'value': threshold, 'exclude': not_sf},
        'no treshold because' : {'value': "no s2s pipeline", 'exclude': is_sf},
        'fmin :': {'value': f_min, 'exclude': False},
        'fmax: ': {'value': f_max, 'exclude': False},
        'n_mels: ': {'value': n_mels, 'exclude': False},
        'speech2spikes pipeline is used': {'value': "" , 'exclude': not_sf},
        'time_step or resolution of the spectogramm in msec: ': {'value': hop_length*1000/sample_rate, 'exclude': False},
        'hop_length: ': {'value': hop_length, 'exclude': False}
    }

    plot_variables(variables) #Make a nice information as figure captions to inform about the figures to plot

    num_neurons=n_mels
    cumsum = True #this values controls only speech2spikes part of the code and usually set to "True"
    if train_set is None: #if data is from database than "train_set" sould be set else sould be set "None"
        print("here ....")
        if waveform is None:#if "waveform" is the input data the "train_set" sould be set to "None". If both set to "None", it is raies an error
            raise ValueError("Either waveform or train_set must be provided.")
        if isinstance(waveform, np.ndarray): # Process the waveform if train_set is not provided
            
            processed_waveform = torch.tensor(waveform).unsqueeze(0)
    else: # Code to execute if train_set is provided
        result_data_points = s2s.get_data_by_label(dataset=train_set, target_label=label) # Retrieve waveform data based on the label
        processed_waveform, *_ = result_data_points[0]

    # Speech2spikes Paper:
    # The raw audio is centered by subtracting an estimate of the bias (in practice, the bias is often small enough to be ignored) and scaled to achieve a peak-to-peak amplitude of 0.5.
    tensor = remove_bias_tesnor(processed_waveform)
    tensor = s2s.scale_tensor_list(tensor,desired_amplitude=0.25,desired_bias=0)
    tensor = s2s.pad_sequence_tensor(tensor)
    if plot==True:
        plot_waveform(tensor.squeeze(0).numpy(), sample_rate=sample_rate, title_label="time domain plot", windows_width=16 if spike_encoding == "hw2" else 10)

    if spectral_feature == "Mel":
        Mel_transform = s2s.Mel_spec_transform(hop_length=hop_length, n_mels=n_mels, f_min=f_min, f_max=f_max, mel_scale='slaney')
        tensor = Mel_transform(tensor)
        if plot==True and spike_encoding!="hw2" and spike_encoding!="hw3":
            # plot_mel_spectrogram(tensor, f_min, f_max,  "Meltransform")
            plot_mel_spectrogram(tensor, f_min, f_max,  "Meltransform", cmap='cividis')
        #Below are the mel to hz and hz to mel calculations done by the software
        # def hz_to_mel_htk(hz):
        #     return 2595 * np.log10(1 + hz / 700)
        
        # def mel_to_hz_htk(mel):
        #     return 700 * (10**(mel / 2595) - 1)
        
        # def hz_to_mel_slaney(hz):
        #     return 1127 * np.log10(1 + hz / 700)
        
        # def mel_to_hz_slaney(mel):
        #     return 700 * (10**(mel / 1127) - 1)
    elif spectral_feature == "MFCC":
        MFCC_transform   = s2s.MFCC_transform(hop_length=hop_length, log_mels=log_spectral_feature, n_mels=n_mels, f_min=f_min, f_max=f_max)
        tensor   = MFCC_transform(tensor)
        # print(tensor.shape)
        if plot==True:
            # plot_mel_spectrogram(tensor, f_min, f_max,  "MFCC transform")
            plot_mel_spectrogram(tensor, f_min, f_max,  "MFFC transform", cmap='cividis')
            
            
    else:
        raise ValueError("Invalid param value. Supported values are 'Mel' and 'MFCC'")    

    if spike_encoding=="step_forward" or spike_encoding == "SFSUM":
        # Each feature in the stack is then centered and scaled using static 
        # estimates of the mean and standard deviation derived from trial runs 
        # for data driven feature extraction: 
        print("Hello SFSUM")
        tensor = standardize_batch(tensor)
        _, pos_sig, pos_neg = s2s.step_forward_encoding(tensor, thr=threshold) 
        # tensor = normalize(tensor)
        _, pos_sig, pos_neg = s2s.step_forward_encoding(tensor, thr=threshold) 
    
        # num_neurons=n_mels
        if cumsum:
            csum = torch.cumsum(tensor, dim=-1)
            _, pos_accum, neg_accum = s2s.step_forward_encoding(csum, thr=threshold)
            num_neurons=num_neurons*4 #only plot function needs this 
        tensor = torch.cat((pos_sig, pos_neg, pos_accum, neg_accum), dim=1) if spike_encoding=="step_forward" else tensor
        if plot==True:
            plot_pipline(tensor,num_neurons)
       
    if spike_encoding=="hw2":
        tensor = scale_tensor(tensor, desired_amplitude=1)
        if plot==True:
            plot_mel_spectrogram(tensor, f_min, f_max,  "normalized Meltransform", windows_width=20, cmap='cividis')
        tensor = gen_impulse_oscillator(tensor,  num_bins=6, num_samples=10, test=False )
        if plot==True:
            # plot_tensor_distribution(tensor, bins=2,title="before scale")
            # print("tensor shape after Mel:",tensor.shape)
            plot_pipline(tensor.unsqueeze(0),num_neurons, windows_width=16)
            # plot_pipline(tensor.squeeze(0),num_neurons)
        
    elif spike_encoding=="hw3":
        tensor =scale_tensor(tensor,desired_amplitude=1, desired_bias=0)
        tensor_after_scale=tensor
        tensor = spikegen.rate(tensor, num_steps=1, gain=1, offset=SpikingGenOffset, time_var_input=False)
        if plot==True:
            # print(f'tensor shape input is: {tensor_after_scale.shape} ans output shape is {tensor.shape}')
            plot_mel_spectrogram(tensor_after_scale, f_min, f_max,  "normalized Meltransform", cmap='cividis')
            plot_pipline(tensor.squeeze(0),num_neurons)
 
    return tensor

def plot_tensor_distribution(tensor, bins=50, title="na"):
    """
    Plot the distribution of values in a PyTorch tensor.

    Parameters:
        tensor (torch.Tensor): The input tensor.
        bins (int): Number of bins for the histogram. Default is 50.
    """
    # Flatten the tensor into a 1D array
    flattened_tensor = tensor.flatten()

    # Convert the tensor to a NumPy array
    numpy_array = flattened_tensor.numpy()

    # Plot a histogram of the values
    plt.hist(numpy_array, bins=bins)
    plt.title(f'Distribution of Tensor Values  {title}')
    plt.xlabel('Amplitude')
    plt.ylabel('Count')
    plt.show()


def standardize_batch(batch):
    if batch.dim() == 4:  # Shape [batch_size, channels, filterbank, time]
        mean = batch.mean(dim=(0, 3), keepdim=True)  # Compute mean along batch and time dimensions
        std = batch.std(dim=(0, 3), keepdim=True)    # Compute std along batch and time dimensions
    elif batch.dim() == 3:  # Shape [channels, filterbank, time]
        mean = batch.mean(dim=(1, 2), keepdim=True)  # Compute mean along filterbank and time dimensions
        std = batch.std(dim=(1, 2), keepdim=True)    # Compute std along filterbank and time dimensions
    else:
        raise ValueError("Batch must have 3 or 4 dimensions.")
    
    standardized_batch = (batch - mean) / std  # Standardize the batch
    return standardized_batch
# # Example usage for shape [64, 1, 20, 201]
# batch_data_4d = torch.randn(64, 1, 20, 201)  # Example batch data
# standardized_batch_data_4d = standardize_batch(batch_data_4d)
# print("Standardized batch data shape [64, 1, 20, 201]:", standardized_batch_data_4d.shape)

# # Example usage for shape [1, 20, 201]
# batch_data_3d = torch.randn(1, 20, 201)  # Example batch data
# standardized_batch_data_3d = standardize_batch(batch_data_3d)
# print("Standardized batch data shape [1, 20, 201]:", standardized_batch_data_3d.shape)

def scale_tensor(tensor, desired_amplitude=0.5,desired_bias=0):
    abs_max = torch.max(torch.abs(tensor))
    scaling_factor = desired_amplitude / abs_max
    tensor = tensor * scaling_factor + desired_bias
    return tensor

# Example usage:
# plot_tensor_distribution(your_tensor)

def linear_interpolation(x1, y1, x2, y2):
    # Calculate the slope (m) and y-intercept (b)
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    
    # Define the linear function and return the y-value
    def interpolate(x):
        return m * x + b
    
    return interpolate
# usage:
# # Example usage
# x1, y1 = 2.0, 3.0  # First point
# x2, y2 = 5.0, 7.0  # Second point

# # Create the interpolation function
# interpolate_func = linear_interpolation(x1, y1, x2, y2)

# # Calculate y for a given x value
# x_value = 4.0  # Example x value
# y_value = interpolate_func(x_value)

# print(f"When x = {x_value}, y = {y_value}")

import matplotlib.pyplot as plt

def plot_mel_spectrogram(mel_spectrogram, f_min, f_max,  title, cmap='plasma', windows_width=13):
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
    

def plot_variables(variables_dict, figsize=(10, 0.1), row_limit=50):
    """
    Plot variable names and values as text annotations in a figure.

    Parameters:
    - variables_dict (dict): Dictionary containing variable names, values, and exclusion flags.
    - figsize (tuple): Figure size in inches (width, height).
    - row_limit (int): Maximum characters per row.

    Returns:
    - None
    """
    # Create an empty figure without axes
    fig = plt.figure(figsize=figsize)
    plt.axis('off')  # Turn off the axes

    # Calculate the horizontal and vertical positions for centering
    horizontal_position = 0.5
    vertical_position = 0.5

    # Text content
    text_content = 'Pipeline settings:\n'
    row_length = 0
    for name, value in variables_dict.items():
        # Check if the variable should be printed based on the exclusion flag
        if value.get('exclude', False):
            continue  # Skip printing this variable
        # Add variable name and value to the row
        text_content += f"{name}  {value['value']} | "
        row_length += len(name) + len(str(value['value'])) + 5  # Length of name + length of value + length of delimiters
        # Check if row length exceeds certain limit (e.g., row_limit characters)
        if row_length > row_limit:
            text_content += '\n'  # Start a new row
            row_length = 0  # Reset row length

    # Add text annotation above the figure
    plt.text(horizontal_position, vertical_position, text_content, fontsize=12, ha='center', va='center', style='italic')  # Change the font style to italic

    # Show the plot
    plt.show()
    
    #usage:
    # # Sample variables with exclusion flags
    # variables = {
    #     'var1': {'value': 42, 'exclude': False},
    #     'var2': {'value': 3.14, 'exclude': False},
    #     'var3': {'value': 'hello', 'exclude': True},  # Excluded variable
    #     'var4': {'value': [1, 2, 3], 'exclude': False},
    #     'var5': {'value': {'a': 1, 'b': 2}, 'exclude': False},
    #     'var6': {'value': True, 'exclude': False},
    #     'var7': {'value': (1, 2, 3), 'exclude': False},
    #     'var8': {'value': None, 'exclude': False},
    #     'var9': {'value': 5 + 7j, 'exclude': False},
    #     'var10': {'value': [True, False, True], 'exclude': False}
    # }
    
    # # Plot the variables
    # plot_variables(variables)
    
    # # Plot the variables
    # plot_variables(variables)


