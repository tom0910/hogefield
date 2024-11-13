# display_functions.py
from core.CustomMelScale import CustomMelScale
from torchaudio.transforms import Spectrogram
from IPython.display import Audio, display
import numpy as np
from display.plotting import plot_waveform, plot_mel_spectrogram, plot_spikes, plot_mel_spectrogram_inv
import utils.functional as FU

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def display_audio_in_widget(audio_sample, output_widget):
    """Display audio playback in the specified widget."""
    waveform, sample_rate = audio_sample.load_waveform()
    with output_widget:
        output_widget.clear_output(wait=True)
        display(Audio(waveform.numpy(), rate=sample_rate))

def display_audio_in_widget_2(waveform, sample_rate, output_widget):
    """Display audio playback in the specified widget."""
    # waveform, sample_rate = audio_sample.load_waveform()
    with output_widget:
        output_widget.clear_output(wait=True)
        display(Audio(waveform.numpy(), rate=sample_rate))

# def plot_audio_waveform_in_widget(audio_sample, output_widget):
#     """Plot audio waveform in the specified widget."""
#     waveform, sample_rate = audio_sample.load_waveform()
#     audio_length_in_sec = waveform.size(0) / sample_rate
#     time = np.linspace(0, audio_length_in_sec, num=waveform.size(0))
#     with output_widget:
#         output_widget.clear_output(wait=True)
#         plot_waveform(time, waveform, title="Audio Amplitude Over Time")


def plot_audio_waveform_in_widget(audio_sample, output_frame):
    # Create the plot
    waveform, sample_rate = audio_sample.load_waveform()
    audio_length_in_sec = waveform.size(0) / sample_rate
    time = np.linspace(0, audio_length_in_sec, num=waveform.size(0))

    """Plot audio waveform in the specified Tkinter frame."""
    # Clear previous content in the frame
    for widget in output_frame.winfo_children():
        widget.destroy()
        plot_waveform(time, waveform, title="Audio Amplitude Over Time")

# def plot_audio_waveform_in_widget(audio_sample, output_frame):
#     """Plot audio waveform in the specified Tkinter frame."""
#     # Clear previous content in the frame
#     for widget in output_frame.winfo_children():
#         widget.destroy()

#     # Create the plot
#     waveform, sample_rate = audio_sample.load_waveform()
#     audio_length_in_sec = waveform.size(0) / sample_rate
#     time = np.linspace(0, audio_length_in_sec, num=waveform.size(0))

#     fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
#     ax.plot(time, waveform)
#     ax.set_title("Audio Amplitude Over Time")

#     # Embed the figure into the Tkinter frame using FigureCanvasTkAgg
#     canvas = FigureCanvasTkAgg(fig, master=output_frame)
#     canvas.draw()
#     canvas_widget = canvas.get_tk_widget()
#     canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def plot_audio_waveform_in_widget_2(waveform, sample_rate, output_widget):
    """Plot audio waveform in the specified widget."""
    # waveform, sample_rate = audio_sample.load_waveform()
    audio_length_in_sec = waveform.size(0) / sample_rate
    time = np.linspace(0, audio_length_in_sec, num=waveform.size(0))
    with output_widget:
        output_widget.clear_output(wait=True)
        plot_waveform(time, waveform, title="Audio Amplitude Over Time")

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

def polt_reverse(audio_sample, mel_config, spikes_data, output_widget, output_widget2, output_widget3):
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
        output_widget=output_widget,
        title="Approximated Mel Spectrogram"
    )
    waveform_approx = FU.inverse_mel_to_waveform(mel_spectrogram_approx, mel_config, sample_rate)
    plot_audio_waveform_in_widget_2(waveform_approx, sample_rate, output_widget2)
    display_audio_in_widget_2(waveform_approx, sample_rate, output_widget3)
    
