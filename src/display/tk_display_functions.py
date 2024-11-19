# display_functions.py
from core.CustomMelScale import CustomMelScale
from torchaudio.transforms import Spectrogram
from IPython.display import Audio, display
import numpy as np
from display.plotting import plot_waveform, plot_mel_spectrogram, plot_spikes, plot_mel_spectrogram_inv
import utils.functional as FU

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from IPython.display import Audio
from tkhtmlview import HTMLLabel
import torch
import matplotlib.pyplot as plt



def xy_waveform_calc(audio_sample):
    """Plot the waveform of the audio file in a Tkinter frame using existing plot_waveform."""
    # Load the waveform from the audio_sample (adjust as per your AudioSample class)
    waveform, sample_rate = audio_sample.load_waveform()
    # Calculate the time axis in seconds
    audio_length_in_sec = waveform.size(0) / sample_rate
    time = np.linspace(0, audio_length_in_sec, num=waveform.size(0))
    
    return time, waveform.numpy()

import os
import soundfile as sf


import os
import soundfile as sf

def save_audio(
    audio_sample=None, 
    waveform=None, 
    sample_rate=None, 
    file_name="original_waveform.wav", 
    save_dir="src/run/SaveAudio"
):
    """
    Saves audio data to a specified directory.

    Args:
        audio_sample (AudioSample, optional): Provides waveform and sample rate when `load_waveform` is called.
        waveform (torch.Tensor or np.ndarray, optional): The waveform to be saved. Required if `audio_sample` is not provided.
        sample_rate (int, optional): The sample rate of the waveform. Required if `audio_sample` is not provided.
        file_name (str, optional): Name of the output audio file. Default is "original_waveform.wav".
        save_dir (str, optional): Directory to save the audio file. Default is "src/run/SaveAudio".

    Raises:
        ValueError: If neither `audio_sample` nor (`waveform` and `sample_rate`) are provided.
    """
    # Case 1: Use `audio_sample` to fetch waveform and sample_rate
    if audio_sample:
        waveform, sample_rate = audio_sample.load_waveform()

    # Case 2: Use provided `waveform` and `sample_rate`
    elif waveform is not None and sample_rate is not None:
        # Ensure the waveform is in NumPy format
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
    else:
        raise ValueError("Either `audio_sample` or both `waveform` and `sample_rate` must be provided.")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)

    # Write the audio to file
    sf.write(file_path, waveform, sample_rate)
    print(f"Audio saved to {file_path}")

def mel_spctgrm_calc(audio_sample, mel_config):
    """Plot the mel spectrogram and filter banks in the specified widget."""
    mel_spectrogram, sample_rate, custom_mel_scale  = FU.get_mel_spectrogram(audio_sample, mel_config)
    mel_spectrogram = normalize_spectrogram_global(mel_spectrogram)

    return(mel_spectrogram, audio_sample, mel_config, sample_rate, custom_mel_scale)

def reconstructed_mel_spctgrm_calc(audio_sample, mel_config):
    """Plot the mel spectrogram and filter banks in the specified widget."""
    # mel_spectrogram, sample_rate, custom_mel_scale  = FU.get_mel_spectrogram(audio_sample, mel_config)
    # mel_spectrogram = normalize_spectrogram_global(mel_spectrogram)

    # return(mel_spectrogram, audio_sample, mel_config, sample_rate, custom_mel_scale)

def normalize_spectrogram_global(mel_spectrogram):
    """Normalize the mel spectrogram globally to a 0-1 range."""
    return (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())

# def polt_reverse(audio_sample, mel_config, spikes_data, canvas1, ax1, canvas2,ax2):
#     threshold = spikes_data.threshold
    
#     # sample_rate = audio_sample.sample_rate
#     # waveform, sample_rate = audio_sample.load_waveform() # lehet, hogy ezt be kell tenni a fuggvbe
#     # num_neurons, num_spike_index, spikes,_,_ = FU.generate_spikes(audio_sample, mel_config, threshold, norm_inp=True, norm_cumsum=True)
#     # mel_spectrogram_approx=FU.inverse_generate_spikes(spikes, mel_config, sample_rate, threshold, norm_inp=True, norm_cumsum=True)
#     # Plot the approximated mel spectrogram
#     plot_mel_spectrogram_inv(
#         mel_spectrogram=mel_spectrogram_approx,
#         audio_sample=audio_sample,
#         mel_config=mel_config,
#         sample_rate=sample_rate,
#         custom_mel_scale=None,  # Or pass if custom_mel_scale is needed
#         title="Approximated Mel Spectrogram",
#         canvas=canvas1,
#         ax = ax1
#     )    
#     # reconstruct and plot audiowave function 
#     waveform_approx = FU.inverse_mel_to_waveform(mel_spectrogram_approx, mel_config, sample_rate)
#     # Convert waveform to numpy if it's a torch.Tensor
#     if isinstance(waveform, torch.Tensor):
#         waveform_approx = waveform_approx.numpy()
#     audio_length_in_sec = waveform_approx.size / sample_rate
#     print(f"wave approx in sec, wave size, srate: {audio_length_in_sec}, {waveform_approx.size}, {sample_rate}")
#     time = np.linspace(0, audio_length_in_sec, num=waveform_approx.size)

        
#     plot_waveform(time, waveform_approx, canvas2, ax2)
#     save_audio(waveform=waveform_approx,sample_rate=sample_rate, file_name="reconstructed_waveform.wav", save_dir="src/run/SaveAudio")
    


    
