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

def display_audio_in_widget(audio_sample, output_widget):
    """Display audio playback in the specified widget."""
    waveform, sample_rate = audio_sample.load_waveform()
    with output_widget:
        output_widget.clear_output(wait=True)
        display(Audio(waveform.numpy(), rate=sample_rate))

# def display_audio_in_widget_2(waveform, sample_rate, output_widget):
#     """Display audio playback in the specified widget."""
#     # waveform, sample_rate = audio_sample.load_waveform()
#     with output_widget:
#         output_widget.clear_output(wait=True)
#         display(Audio(waveform.numpy(), rate=sample_rate))

# import simpleaudio as sa
# import numpy as np
# import tkinter as tk

# def display_audio_in_widget_2(waveform, sample_rate, output_frame):
#     """
#     Display audio playback in a Tkinter frame with a playback button.

#     Args:
#         waveform (np.ndarray or torch.Tensor): The audio waveform data.
#         sample_rate (int): The sample rate of the audio.
#         output_frame (tk.Frame): The Tkinter frame to display the audio player.
#     """
#     # Clear any existing content in the frame
#     for widget in output_frame.winfo_children():
#         widget.destroy()

#     # Convert waveform to numpy if it's a torch.Tensor
#     if isinstance(waveform, torch.Tensor):
#         waveform = waveform.numpy()

#     # Scale waveform to 16-bit PCM format
#     audio_data = (waveform * 32767).astype(np.int16)

#     # Function to play the audio
#     def play_audio():
#         play_obj = sa.play_buffer(audio_data, 1, 2, sample_rate)
#         play_obj.wait_done()

#     # Create a playback button in the Tkinter frame
#     play_button = tk.Button(output_frame, text="Play Audio", command=play_audio)
#     play_button.pack(pady=10)


def display_audio_in_widget_2(waveform, sample_rate, output_frame):
    """
    Display audio playback in a Tkinter frame using an HTML audio widget.

    Args:
        waveform (np.ndarray or torch.Tensor): The audio waveform data.
        sample_rate (int): The sample rate of the audio.
        output_frame (tk.Frame): The Tkinter frame to display the audio player.
    """
    # Clear any existing content in the frame
    for widget in output_frame.winfo_children():
        widget.destroy()

    # Convert waveform to numpy if it's a torch.Tensor
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    # Generate an IPython Audio widget
    audio_widget = Audio(waveform, rate=sample_rate)
    audio_html = audio_widget._repr_html_()

    # Embed the HTML audio player into the Tkinter frame using tkhtmlview
    html_label = HTMLLabel(output_frame, html=audio_html)
    html_label.pack(fill="both", expand=True)


# def plot_audio_waveform_in_widget(audio_sample, output_widget):
#     """Plot audio waveform in the specified widget."""
#     waveform, sample_rate = audio_sample.load_waveform()
#     audio_length_in_sec = waveform.size(0) / sample_rate
#     time = np.linspace(0, audio_length_in_sec, num=waveform.size(0))
#     with output_widget:
#         output_widget.clear_output(wait=True)
#         plot_waveform(time, waveform, title="Audio Amplitude Over Time")

import matplotlib.pyplot as plt

def plot_audio_waveform_in_widget(audio_sample, output_frame):
    """Plot the waveform of the audio file in a Tkinter frame using existing plot_waveform."""
    
    # Load the waveform from the audio_sample (adjust as per your AudioSample class)
    waveform, sample_rate = audio_sample.load_waveform()
    
    # Calculate the time axis in seconds
    audio_length_in_sec = waveform.size(0) / sample_rate
    time = np.linspace(0, audio_length_in_sec, num=waveform.size(0))

    plot_waveform(output_frame, time, waveform.numpy(), title="Audio Amplitude Over Time") 


# def plot_audio_waveform_in_widget_2(waveform, sample_rate, output_widget):
#     """Plot audio waveform in the specified widget."""
#     # waveform, sample_rate = audio_sample.load_waveform()
#     audio_length_in_sec = waveform.size(0) / sample_rate
#     time = np.linspace(0, audio_length_in_sec, num=waveform.size(0))
#     with output_widget:
#         output_widget.clear_output(wait=True)
#         plot_waveform(time, waveform, title="Audio Amplitude Over Time")
        

def plot_audio_waveform_in_widget_2(waveform, sample_rate, output_frame):
    """
    Plot audio waveform in a Tkinter frame.

    Args:
        waveform (np.ndarray or torch.Tensor): The audio waveform data.
        sample_rate (int): The sample rate of the audio.
        output_frame (tk.Frame): The Tkinter frame to display the plot.
    """
    # Clear any existing content in the frame
    for widget in output_frame.winfo_children():
        widget.destroy()

    # Convert waveform to numpy if it's a torch.Tensor
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    # Calculate the time axis in seconds
    audio_length_in_sec = waveform.size / sample_rate
    time = np.linspace(0, audio_length_in_sec, num=waveform.size)

    # Create the Matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the waveform
    ax.plot(time, waveform, color="blue", linewidth=0.8)
    ax.set_title("Audio Amplitude Over Time", fontsize=12)
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)

    # Embed the Matplotlib figure into the Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=output_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
        

def normalize_spectrogram_global(mel_spectrogram):
    """Normalize the mel spectrogram globally to a 0-1 range."""
    return (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())

def plot_mel_spectrogram_in_widget(audio_sample, mel_config, output_widget):
    """Plot the mel spectrogram and filter banks in the specified widget."""
    mel_spectrogram, sample_rate, custom_mel_scale  = FU.get_mel_spectrogram(audio_sample, mel_config)
    mel_spectrogram = normalize_spectrogram_global(mel_spectrogram)

    # Plot mel spectrogram
    plot_mel_spectrogram(mel_spectrogram, audio_sample, mel_config, sample_rate, custom_mel_scale, output_widget)


def plot_spikes_in_widget(audio_sample, mel_config, spikes_data, plot_radio, output_widget):    
    plot_spikes(audio_sample, mel_config, spikes_data, plot_radio, output_widget)

def polt_reverse(audio_sample, mel_config, spikes_data, output_frame, output_frame2, output_frame3):
    threshold = spikes_data.threshold
    num_neurons, num_spike_index, spikes,_,_ = FU.generate_spikes(audio_sample, mel_config, threshold, norm_inp=True, norm_cumsum=True)
    # sample_rate = audio_sample.sample_rate
    waveform, sample_rate = audio_sample.load_waveform() # lehet, hogy ezt be kell tenni a fuggvbe
    mel_spectrogram_approx=FU.inverse_generate_spikes(spikes, mel_config, sample_rate, threshold, norm_inp=True, norm_cumsum=True)
    # Plot the approximated mel spectrogram
    plot_mel_spectrogram_inv(
        mel_spectrogram=mel_spectrogram_approx,
        audio_sample=audio_sample,
        mel_config=mel_config,
        sample_rate=sample_rate,
        custom_mel_scale=None,  # Or pass if custom_mel_scale is needed
        output_frame=output_frame,
        title="Approximated Mel Spectrogram"
    )
    waveform_approx = FU.inverse_mel_to_waveform(mel_spectrogram_approx, mel_config, sample_rate)
    plot_audio_waveform_in_widget_2(waveform_approx, sample_rate, output_frame2)
    display_audio_in_widget_2(waveform_approx, sample_rate, output_frame3)
    
# Create the plots frame without embedding the waveform plot
def create_plots_frame(root):
    
    # Create a frame for widgets (left side)
    widget_frame = ttk.Frame(root, padding=10, relief="ridge")
    widget_frame.grid(row=0, column=0, sticky="ns")  # Left-side frame
        
    # Create a frame for the plots
    plot_frame = ttk.Frame(root, padding=10, relief="sunken")
    plot_frame.grid(row=0, column=1, sticky="nsew")

    # Define 2x3 grid
    subframes = [ttk.Frame(plot_frame) for _ in range(6)]
    for idx, frame in enumerate(subframes):
        row, col = divmod(idx, 3)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        
    # Set grid weights for dynamic resizing
    for i in range(2):  # Two rows
        plot_frame.rowconfigure(i, weight=1)
    for i in range(3):  # Three columns
        plot_frame.columnconfigure(i, weight=1)        
    
    out_audio_wave_plt = subframes[0]  
    out_mel_sptgrm_plt = subframes[1]  
    out_spike_raster_plt = subframes[2]
    out_rev_spike_raster_plt = subframes[3]
    out_rev_mel_sptgrm_plt = subframes[4]  
    out_rev_play = subframes[5] 
    return widget_frame, out_audio_wave_plt, out_mel_sptgrm_plt, out_spike_raster_plt, out_rev_spike_raster_plt, out_rev_mel_sptgrm_plt, out_rev_play  # Return the first subframe for the waveform plot


    
