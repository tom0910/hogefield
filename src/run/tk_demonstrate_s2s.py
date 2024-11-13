# tk_demonstrate_s2s.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio.transforms import AmplitudeToDB, GriffinLim
import torchaudio.transforms as T
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Ensure src is in PYTHONPATH
# sys.path.insert(0, os.path.abspath('../src'))

# Project imports (assuming src is in PYTHONPATH)
from core.AudioSample import AudioSample
from core.MelSpectrogramConfig import MelSpectrogramConfig
import config.config as C
from config.config import BASE_PATH, DEFAULT_DIRECTORY, DEFAULT_FILE_INDEX, DEFAULT_THRESHOLD
from display.tk_widgets_setup import create_widgets
from display.tk_display_functions import (
    display_audio_in_widget, plot_audio_waveform_in_widget,
    plot_mel_spectrogram_in_widget, plot_spikes_in_widget, polt_reverse
)
from utils.widget_sync_utils import set_audio_sample_from_widget_values, set_mel_config_from_widget_values
from core.Spikes import Spikes

# Create the main Tkinter window
root = tk.Tk()
root.title("Spie Generation Sensory Lab")
root.geometry("1400x800")

# Initialize AudioSample and MelSpectrogramConfig
audio_sample = AudioSample(BASE_PATH, DEFAULT_DIRECTORY, DEFAULT_FILE_INDEX)
mel_config = MelSpectrogramConfig(
    audio_sample,
    n_fft=C.DEFAULT_N_FFT,
    hop_length=C.DEFAULT_HOP_LENGTH,
    n_mels=C.DEFAULT_N_MELS,
    f_min=C.DEFAULT_F_MIN,
    f_max=C.DEFAULT_F_MAX,
    power=C.DEFAULT_POWER,
    filter_type="custom",
    toggle_mel_filter="spktgrm"
)
spikes_data = Spikes(threshold=C.DEFAULT_THRESHOLD)

# Create Widgets using Tkinter
(
    directory_dropdown,
    file_slider,
    n_fft_input,
    hop_length_slider,
    n_mels_slider,
    f_min_slider,
    f_max_slider,
    power_toggle,
    threshold_slider,
    spike_plot_radio,
    spike_periode_slider,
    spk_freq_label,
    channel_slider,
    filter_choice_radio,
    mel_plot_radio,
    spk_freq_label
) = create_widgets(audio_sample, mel_config, spikes_data, root)

# output widget or frame 
# output_audio_signal = ttk.Frame(root)
# output_audio_signal.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# Ensure the frame expands to fill available space
# root.grid_columnconfigure(0, weight=1)
# root.grid_rowconfigure(1, weight=1)

# Observers for widget changes

def update_plot():
    set_audio_sample_from_widget_values(audio_sample, directory_dropdown, file_slider)
    # plot_audio_waveform_in_widget(audio_sample, output_audio_signal)
    update_plot_mel()
    polt_reverse(audio_sample, mel_config, spikes_data, output_rev_spikes, output_rev_mel, output_rev_play)


# def update_plot():
#     set_audio_sample_from_widget_values(audio_sample, directory_dropdown, file_slider)
#     plot_audio_waveform_in_widget(audio_sample, output_audio_signal)
#     update_plot_mel()
#     polt_reverse(audio_sample, mel_config, spikes_data, output_rev_spikes, output_rev_mel, output_rev_play)


def update_plot_mel():
    set_mel_config_from_widget_values(
        mel_config, n_fft_input, hop_length_slider, n_mels_slider,
        f_min_slider, f_max_slider, power_toggle, filter_choice_radio, mel_plot_radio
    )
    plot_mel_spectrogram_in_widget(audio_sample, mel_config, output_melspecrogram)
    update_plot_spike()
    channel_slider.config(to=mel_config.n_mels)
    polt_reverse(audio_sample, mel_config, spikes_data, output_rev_spikes, output_rev_mel, output_rev_play)


def update_plot_spike():
    spikes_data.threshold = float(threshold_slider.get())
    plot_radio = spike_plot_radio.get()
    plot_spikes_in_widget(audio_sample, mel_config, spikes_data, plot_radio, output_spikes)
    polt_reverse(audio_sample, mel_config, spikes_data, output_rev_spikes, output_rev_mel, output_rev_play)


def check_filter_type():
    """Set widget values and disable state based on filter choice."""
    if filter_choice_radio.get() == "custom":
        f_min_slider.set(275)
        f_max_slider.set(7625)
        n_mels_slider.set(16)
        f_min_slider.config(state=tk.DISABLED)
        f_max_slider.config(state=tk.DISABLED)
        n_mels_slider.config(state=tk.DISABLED)
    else:
        f_min_slider.config(state=tk.NORMAL)
        f_max_slider.config(state=tk.NORMAL)
        n_mels_slider.config(state=tk.NORMAL)
    update_plot_mel()

# Button to update plot
# update_button = ttk.Button(root, text="Update Plot", command=update_plot)
# update_button.grid(row=13, column=0, pady=5)

# # Create a Figure and Plot in the Figure Frame
# x = np.linspace(0, 10, 100)
# fig = plt.Figure(figsize=(5, 4), dpi=100)
# ax = fig.add_subplot(111)
# ax.plot(x, np.sin(x))

# # Embed the figure to the Tkinter window
# canvas = FigureCanvasTkAgg(fig, master=root)
# canvas.draw()
# canvas_widget = canvas.get_tk_widget()
# canvas_widget.grid(row=0, column=1, padx=10, pady=10)

# Run the main Tkinter event loop
root.mainloop()
