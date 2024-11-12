import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio.transforms import AmplitudeToDB, GriffinLim
import torchaudio.transforms as T
from IPython.display import Audio, display
import ipywidgets as widgets
from ipywidgets import GridspecLayout

# Ensure src is in PYTHONPATH
sys.path.insert(0, os.path.abspath('../src'))

# Project imports (assuming src is in PYTHONPATH)
from core.AudioSample import AudioSample
from core.MelSpectrogramConfig import MelSpectrogramConfig
import config.config as C
from config.config import BASE_PATH, DEFAULT_DIRECTORY, DEFAULT_FILE_INDEX, DEFAULT_THRESHOLD
from display.widgets_setup import create_widgets
from display.display_functions import (
    display_audio_in_widget, plot_audio_waveform_in_widget,
    plot_mel_spectrogram_in_widget, plot_spikes_in_widget, polt_reverse
)
from utils.widget_sync_utils import set_audio_sample_from_widget_values, set_mel_config_from_widget_values
from core.Spikes import Spikes

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

# Initialize widgets
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
    spike_period_slider_combo,
    spk_freq_label,
    channel_slider,
    filter_choice_radio,
    mel_plot_radio,
    spk_freq_label
) = create_widgets(audio_sample, mel_config, spikes_data)

# Output widgets
output_play = widgets.Output(layout=widgets.Layout())
output_audio_signal = widgets.Output(layout=widgets.Layout())
output_melspecrogram = widgets.Output(layout=widgets.Layout())
output_spikes = widgets.Output(layout=widgets.Layout())
output_rev_spikes = widgets.Output(layout=widgets.Layout())
output_rev_mel = widgets.Output(layout=widgets.Layout())
output_rev_play = widgets.Output(layout=widgets.Layout())

# Function for testing
def beep_sound(message):
    print(f'Beep function called {message}')
    frequency = 440  # Hz
    duration = 0.2   # seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    display(Audio(wave, rate=sample_rate, autoplay=True))

# Observers for widget changes
def update_plot(change):
    set_audio_sample_from_widget_values(audio_sample, directory_dropdown, file_slider)
    display_audio_in_widget(audio_sample, output_play)
    plot_audio_waveform_in_widget(audio_sample, output_audio_signal)
    update_plot_mel(None)
    polt_reverse(audio_sample, mel_config, spikes_data, output_rev_spikes, output_rev_mel, output_rev_play)

def update_plot_mel(change):
    set_mel_config_from_widget_values(
        mel_config, n_fft_input, hop_length_slider, n_mels_slider,
        f_min_slider, f_max_slider, power_toggle, filter_choice_radio, mel_plot_radio
    )
    plot_mel_spectrogram_in_widget(audio_sample, mel_config, output_melspecrogram)
    update_plot_spike(None)
    channel_slider.max = mel_config.n_mels
    polt_reverse(audio_sample, mel_config, spikes_data, output_rev_spikes, output_rev_mel, output_rev_play)

def update_plot_spike(change):
    spikes_data.threshold = threshold_slider.value
    plot_radio = spike_plot_radio.value
    plot_spikes_in_widget(audio_sample, mel_config, spikes_data, plot_radio, output_spikes)
    polt_reverse(audio_sample, mel_config, spikes_data, output_rev_spikes, output_rev_mel, output_rev_play)

def check_filter_type(change):
    """Set widget values and disable state based on filter choice."""
    if filter_choice_radio.value == "custom":
        f_min_slider.value = 275
        f_max_slider.value = 7625
        n_mels_slider.value = 16
        f_min_slider.disabled = True
        f_max_slider.disabled = True
        n_mels_slider.disabled = True
    else:
        f_min_slider.disabled = False
        f_max_slider.disabled = False
        n_mels_slider.disabled = False
    update_plot_mel(change)

# Attach observers
directory_dropdown.observe(update_plot, names='value')
file_slider.observe(update_plot, names='value')
n_fft_input.observe(update_plot_mel, names='value')
hop_length_slider.observe(update_plot_mel, names='value')
n_mels_slider.observe(update_plot_mel, names='value')
f_min_slider.observe(update_plot_mel, names='value')
f_max_slider.observe(update_plot_mel, names='value')
power_toggle.observe(update_plot_mel, names='value')
threshold_slider.observe(update_plot_spike, names='value')
spike_plot_radio.observe(update_plot_spike, names='value')
filter_choice_radio.observe(check_filter_type, names='value')
mel_plot_radio.observe(update_plot_mel, names='value')

# Initial display setup
update_plot(None)
update_plot_mel(None)
update_plot_spike(None)

# Widget layout and display setup
centered_layout = widgets.Layout(align_items='center', justify_content='center')
align_top_layout = widgets.Layout(align_items='flex-start', justify_content='flex-start')

file_controls = widgets.VBox([directory_dropdown, file_slider, output_play], layout=centered_layout)
mel_spectrogram_controls = widgets.VBox(
    [hop_length_slider, n_mels_slider, f_min_slider, f_max_slider, n_fft_input, power_toggle, filter_choice_radio, mel_plot_radio],
    layout=centered_layout
)
spike_control = widgets.VBox([threshold_slider, spike_plot_radio, spike_period_slider_combo], layout=centered_layout)

audio_output = widgets.VBox([output_audio_signal], layout=align_top_layout)
mel_spectrogram_output = widgets.VBox([output_melspecrogram], layout=align_top_layout)
spike_output = widgets.VBox([output_spikes], layout=align_top_layout)

inv_audio_output = widgets.VBox([output_rev_mel], layout=align_top_layout)
inv_mel_spectrogram_output = widgets.VBox([output_rev_spikes], layout=align_top_layout)
inv_play_output = widgets.VBox([output_rev_play], layout=align_top_layout)

debug_output = widgets.VBox([spk_freq_label], layout=align_top_layout)

control_layout = widgets.HBox([file_controls, mel_spectrogram_controls, spike_control], layout=widgets.Layout(justify_content='space-around'))
figure_layout = widgets.HBox([audio_output, mel_spectrogram_output, spike_output], layout=widgets.Layout(justify_content='space-around'))
figure_layout_2 = widgets.HBox([inv_audio_output, inv_mel_spectrogram_output, inv_play_output], layout=widgets.Layout(justify_content='space-around'))
figure_layout_3 = widgets.HBox([debug_output], layout=widgets.Layout(justify_content='space-around'))

display(control_layout)
display(figure_layout)
display(figure_layout_2)
display(figure_layout_3)
