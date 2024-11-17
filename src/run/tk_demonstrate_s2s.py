# tk_demonstrate_s2s.py
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
from display.tk_widgets_setup import create_widgets, initialize_plots, create_plots_frame
from display.tk_display_functions import ( xy_waveform_calc, save_audio, mel_spctgrm_calc,
    plot_spikes_in_widget, polt_reverse)
from utils.widget_sync_utils import set_audio_sample_from_widget_values, set_mel_config_from_widget_values
from core.Spikes import Spikes
from display.plotting import plot_waveform, plot_mel_spectrogram, plot_spikes, plot_mel_spectrogram_inv

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
    filter_type=C.DEFAULT_FILTER_CHOICE,
    toggle_mel_filter="spktgrm"
)
spikes_data = Spikes(threshold=C.DEFAULT_THRESHOLD)

# Create the main Tkinter window
root = tk.Tk()
root.title("Spie Generation Sensory Lab")
root.geometry("2800x800")

# Configure root grid for responsive layout
root.columnconfigure(0, weight=0)  # Left frame fixed size
root.columnconfigure(1, weight=1)  # Right frame expandable
root.rowconfigure(0, weight=1)     # Make the row expandable

# create plots
widget_frame, out_frame_audio_wave_plt, out_frame_mel_sptgrm_plt, out_spike_raster_plt, out_rev_spike_raster_plt, out_rev_mel_sptgrm_plt,out_rev_play = create_plots_frame(root)
# initialize_plots 
frames = [
    out_frame_audio_wave_plt,
    out_frame_mel_sptgrm_plt,
    out_spike_raster_plt,
    out_rev_spike_raster_plt,
    out_rev_mel_sptgrm_plt
]

figs, axes, canvases = initialize_plots(frames)


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
    filter_choice_widget,
    mel_filter_plot_radio_widget,
    # mel_plot_radio,
    spk_freq_label,
    hop_length_entry,
    n_mels_entry
) = create_widgets(audio_sample, mel_config, spikes_data, widget_frame)

# Observers for widget changes
def update_plot():
    set_audio_sample_from_widget_values(audio_sample, directory_dropdown, file_slider)
    save_audio(audio_sample, file_name="original_waveform.wav", save_dir="src/run/SaveAudio")
    time, amplitude = xy_waveform_calc(audio_sample)
    plot_waveform(time, amplitude, canvases[0], axes[0])
    update_plot_mel()
    # polt_reverse(audio_sample, mel_config, spikes_data, out_rev_spike_raster_plt, out_rev_mel_sptgrm_plt, out_rev_play)

def update_plot_mel():
    print("update_plot_mel triggered")
    set_mel_config_from_widget_values(
        mel_config, n_fft_input, hop_length_slider, n_mels_slider, f_min_slider, f_max_slider, 
        power_toggle, 
        filter_choice_widget,
        # filter_choice_radio,
        mel_filter_plot_radio_widget 
        # mel_plot_radio
    )
    mel_spectrogram, updated_audio_sample, updated_mel_config, sample_rate, custom_mel_scale = mel_spctgrm_calc(audio_sample, mel_config)
    plot_mel_spectrogram(mel_spectrogram, updated_audio_sample, updated_mel_config, sample_rate, custom_mel_scale, canvases[1], axes[1])
   
    # plot_mel_spectrogram_in_widget(audio_sample, mel_config, out_frame_mel_sptgrm_plt)
    # update_plot_spike()
    # channel_slider.config(to=mel_config.n_mels)
    # polt_reverse(audio_sample, mel_config, spikes_data, out_rev_spike_raster_plt, out_rev_mel_sptgrm_plt, out_rev_play)


def update_plot_spike():
    spikes_data.threshold = float(threshold_slider.get())
    plot_radio = spike_plot_radio.get()
    plot_spikes_in_widget(audio_sample, mel_config, spikes_data, plot_radio, out_spike_raster_plt)
    polt_reverse(audio_sample, mel_config, spikes_data, out_rev_spike_raster_plt, out_rev_mel_sptgrm_plt, out_rev_play)


def check_filter_type():
    """Set widget values and disable state based on filter choice."""
    print("check_filter_type triggered")
    if filter_choice_widget.get() == "custom":
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
    
# Attach observers to widgets

directory_dropdown.bind("<<ComboboxSelected>>", lambda e: update_plot())
file_slider.config(command=lambda v: update_plot())
# observ nfft entry
n_fft_input.bind('<Return>', lambda event: update_plot_mel())
# observ hop length entry
hop_length_slider.config(command=lambda v: update_plot_mel())
hop_length_entry.bind('<Return>', lambda event: update_plot_mel())
# observ number of mels entry
n_mels_slider.config(command=lambda v: update_plot_mel())
n_mels_entry.bind('<Return>',lambda event: update_plot_mel())
# setting betwen power and magnitude
power_toggle.bind(update_plot_mel)
# observer that should change filter type in sptgrm
filter_choice_widget.bind(check_filter_type)
# observer to change canvas btwn spectrogram and filter type used 
mel_filter_plot_radio_widget.bind(update_plot_mel)

f_min_slider.config(command=lambda v: update_plot_mel())
f_max_slider.config(command=lambda v: update_plot_mel())
threshold_slider.config(command=lambda v: update_plot_spike())

    

# Initial display setup
update_plot()

# Run the main Tkinter event loop
root.mainloop()
