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
from display.tk_display_functions import xy_waveform_calc, save_audio, mel_spctgrm_calc
from utils.widget_sync_utils import set_audio_sample_from_widget_values, set_mel_config_from_widget_values
from core.Spikes import Spikes
from display.plotting import plot_waveform, plot_mel_spectrogram, plot_spikes, plot_distribution, plot_mel_spectrogram_inv
import utils.functional as FU
import utils.training_functional as TF

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
widget_frame, out_frame_audio_wave_plt, out_frame_mel_sptgrm_plt, out_spike_raster_plt,  out_rev_spike_raster_plt, out_rev_mel_sptgrm_plt, empty = create_plots_frame(root)

# initialize_plots 
frames = [
    out_frame_audio_wave_plt,
    out_frame_mel_sptgrm_plt,
    out_spike_raster_plt,
    empty,
    out_rev_spike_raster_plt,
    out_rev_mel_sptgrm_plt
]

figs, axes, canvases = initialize_plots(frames)


# Create Widgets using Tkinter
(
    directory_dropdown, 
    file_slider, 
    # n_fft_input, 
    n_fft_label, n_fft_slider, n_fft_entry,
    hop_length_slider, 
    n_mels_slider, 
    f_min_slider, 
    f_max_slider, 
    power_toggle, 
    threshold_slider, 
    spike_plot_radio_widget, 
    threshold_entry,
    spike_periode_slider,spk_freq_label,channel_slider, 
    filter_choice_widget,
    mel_filter_plot_radio_widget, 
    spk_freq_label,
    hop_length_entry, n_mels_entry,
    save_button_widget,
    filename_entry      
) = create_widgets(audio_sample, mel_config, spikes_data, widget_frame)

# Observers for widget changes
def update_plot():
    print("update_plot() triggered")
    #objects updates width widgets' values
    set_audio_sample_from_widget_values(audio_sample, directory_dropdown, file_slider)
    #save audio to listen back original database wav file
    save_audio(audio_sample, file_name="original_waveform.wav", save_dir="src/run/SaveAudio")
    #plot waveform
    time, amplitude = xy_waveform_calc(audio_sample)
    plot_waveform(time, amplitude, canvases[0], axes[0])
    # because of new audiosample, change mel spectrogram plot
    update_plot_mel()
    
def update_plot_mel():
    print("update_melspctgrm() triggered")
    #objects updates width widgets' values
    set_mel_config_from_widget_values(
        mel_config, 
        # n_fft_input,
        n_fft_slider, 
        hop_length_slider, n_mels_slider, f_min_slider, f_max_slider, 
        power_toggle, 
        filter_choice_widget,
        mel_filter_plot_radio_widget
    )
    #calculations:
    mel_spectrogram, updated_audio_sample, updated_mel_config, sample_rate, custom_mel_scale = mel_spctgrm_calc(audio_sample, mel_config)
    #plot mel spectrogram
    plot_mel_spectrogram(mel_spectrogram, updated_audio_sample, updated_mel_config, sample_rate, custom_mel_scale, canvases[1], axes[1])
    #because of a new mel spectrogram, change spike train plot
    update_plot_spike()

def update_plot_spike():
    print("update spike plot tiggered")
    #objects updates width widgets' values
    spikes_data.threshold = float(threshold_slider.get())
    spike_threshold_updated = spikes_data.threshold
    plot_choice = spike_plot_radio_widget.get()
    
    #calculation:
    num_neurons, num_spike_index, spikes, _, _ = FU.generate_spikes(audio_sample, mel_config, spike_threshold_updated, norm_inp=False, norm_cumsum=False)
    # num_neurons, num_spike_index, spikes, _, _ = FU.generate_spikes_from_audio(audio_sample, mel_config, spike_threshold_updated, norm_inp=False, norm_cumsum=False)
    print("spikes calculated")
    # plot spike trains or distribution
    if plot_choice == C.DEFAULT_SPIKE_PLT_PICK:
        #plot:
        plot_spikes(spikes=spikes, num_neurons=num_neurons, num_spike_index=num_spike_index, canvas=canvases[2], ax=axes[2])
        print("spikes plotted")
    elif plot_choice == C.DEFAULT_DIST_PLT_PICK:
        #calculations:
        mel_spectrogram, _, _ = FU.get_mel_spectrogram(audio_sample, mel_config)
        mel_spectrogram_noramlized = FU.normalize(mel_spectrogram, normalize_to=1)
        #plot:
        plot_distribution(mel_spectrogram_noramlized, canvases[2], axes[2])
    else:
        raise ValueError("Invalid plot_choice value. Use 'spikes' or 'distribution'.")
    print("reconstructed plots tiggered")
    #calculation of melspectrogram reconstruction:
    mel_spectrogram_approx=FU.inverse_generate_spikes(spikes, mel_config, audio_sample.sample_rate, spike_threshold_updated, norm_inp=True, norm_cumsum=True)
    print("melspctgrm reconstructed")
    #plot:
    plot_mel_spectrogram_inv(
        mel_spectrogram=mel_spectrogram_approx,
        audio_sample=audio_sample,
        hop_length=mel_config.hop_length,
        sample_rate=audio_sample.sample_rate,
        custom_mel_scale=None,  # Or pass if custom_mel_scale is needed
        title="Approximated Mel Spectrogram",
        canvas=canvases[4],
        ax = axes[4]
    )    
    #calculation of audio waveform reconstruction:
    waveform_approx = FU.inverse_mel_to_waveform(mel_spectrogram=mel_spectrogram_approx, mel_config=mel_config , sample_rate=audio_sample.sample_rate)
    # Convert waveform to numpy if it's a torch.Tensor
    if isinstance(waveform_approx, torch.Tensor):
        waveform_approx = waveform_approx.numpy()
    audio_length_in_sec = waveform_approx.size / mel_config.sample_rate
    print(f"wave approx in sec, wave size, srate: {audio_length_in_sec}, {waveform_approx.size}, {mel_config.sample_rate}")
    time = np.linspace(0, audio_length_in_sec, num=waveform_approx.size)
    print("audio waveform data reconstructed")
    #plot waveform
    plot_waveform(time, waveform_approx, canvases[5], axes[5])
    # save audio file to listen
    save_audio(waveform=waveform_approx,sample_rate=mel_config.sample_rate, file_name="reconstructed_waveform.wav", save_dir="src/run/SaveAudio")


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
        n_mels_entry.config(state=tk.DISABLED)
    else:
        f_min_slider.config(state=tk.NORMAL)
        f_max_slider.config(state=tk.NORMAL)
        n_mels_slider.config(state=tk.NORMAL)
    update_plot_mel()
    
# Attach observers to widgets

directory_dropdown.bind("<<ComboboxSelected>>", lambda e: update_plot())
file_slider.config(command=lambda v: update_plot())
# observ nfft entry
# n_fft_input.bind('<Return>', lambda event: update_plot_mel())
n_fft_slider.config(command=lambda v: update_plot_mel())
n_fft_entry.bind('<Return>', lambda event: update_plot_mel())
# observ hop length entry
hop_length_slider.config(command=lambda v: update_plot_mel())
hop_length_entry.bind('<Return>', lambda event: update_plot_mel())
# observ number of mels entry
n_mels_slider.config(command=lambda v: update_plot_mel())
n_mels_entry.bind('<Return>',lambda event: update_plot_mel())
# observer max and min frequencies
f_min_slider.config(command=lambda v: update_plot_mel())
f_max_slider.config(command=lambda v: update_plot_mel())
# setting betwen power and magnitude
power_toggle.bind(update_plot_mel)
# observer that should change filter type in sptgrm
filter_choice_widget.bind(check_filter_type)
# observer to change canvas btwn spectrogram and filter type used 
mel_filter_plot_radio_widget.bind(update_plot_mel)

# observ spike related chages:
threshold_slider.config(command=lambda v: update_plot_spike())
threshold_entry.bind('<Return>', lambda event: update_plot_spike())
spike_plot_radio_widget.bind(update_plot_spike)

#save button observer - save these params. for thraining
def get_params(spikes_data, mel_config, audio_sample):
    """
    Generate the parameters dictionary dynamically based on the current state.

    Parameters:
    - spikes_data: Object containing spike threshold.
    - mel_config: Object containing mel-spectrogram configuration.
    - audio_sample: Object containing audio sample information.

    Returns:
    - dict: The dynamically generated parameters.
    """
    return {
        "batch_size": 128,
        "sf_threshold": spikes_data.threshold,
        "hop_length": mel_config.hop_length,
        "f_min": mel_config.f_min,
        "f_max": mel_config.f_max,
        "n_mels": mel_config.n_mels,
        "n_fft": mel_config.n_fft,
        "wav_file_samples": audio_sample.sample_rate,
        "timestep": TF.calculate_num_frames(
            L=audio_sample.sample_rate,
            n_fft=mel_config.n_fft,
            hop_length=mel_config.hop_length,
            center=True,
            show=True
        )
    }
params = {
    "batch_size": 128,
    "sf_threshold": spikes_data.threshold,
    "hop_length": mel_config.hop_length,
    "f_min": mel_config.f_max,
    "f_max": mel_config.f_min,
    "n_mels": mel_config.n_mels,
    "n_fft": mel_config.n_fft,
    "wav_file_samples": audio_sample.sample_rate,
    "timestep": TF.calculate_num_frames(L=audio_sample.sample_rate, n_fft=mel_config.n_fft, hop_length=mel_config.hop_length, center=True, show=True)
}
save_button_widget.bind("<Button-1>", lambda event: FU.on_click_save_params(params=get_params(spikes_data=spikes_data,mel_config=mel_config,audio_sample=audio_sample), ID=filename_entry.get()))

# Initial display setup
update_plot()

# Run the main Tkinter event loop
root.mainloop()
