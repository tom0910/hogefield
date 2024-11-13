import tkinter as tk
from tkinter import ttk
import config.config as config
import utils.functional as FU

# Function to set up widgets using Tkinter
def create_widgets(audio_sample, mel_config, spikes_data, root):
    print("create_widgets function loaded")
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Tkinter Widgets Setup Example")
    root.geometry("1200x800")

    # Directory Dropdown (Combobox in Tkinter)
    directory_dropdown = ttk.Combobox(root, values=audio_sample.get_directories())
    directory_dropdown.set(config.DEFAULT_DIRECTORY)
    directory_dropdown_label = ttk.Label(root, text='Directory:')

    # File Index Slider
    file_slider_label = ttk.Label(root, text='File Index:')
    file_slider = tk.Scale(root, from_=0, to=len(audio_sample.get_files()) - 1, orient='horizontal')

    # FFT Size Entry
    n_fft_label = ttk.Label(root, text='n_fft:')
    n_fft_input = ttk.Entry(root)
    n_fft_input.insert(0, str(config.DEFAULT_N_FFT))

    # Hop Length Slider
    hop_length_label = ttk.Label(root, text='Hop Length:')
    hop_length_slider = tk.Scale(root, from_=1, to=512, orient='horizontal')
    hop_length_slider.set(config.DEFAULT_HOP_LENGTH)

    # Number of Mel Bands Slider
    n_mels_label = ttk.Label(root, text='n_mels:')
    n_mels_slider = tk.Scale(root, from_=2, to=22, orient='horizontal')
    n_mels_slider.set(config.DEFAULT_N_MELS)

    # f_min Slider
    f_min_label = ttk.Label(root, text='f_min:')
    f_min_slider = tk.Scale(root, from_=0, to=8000, resolution=100, orient='horizontal')
    f_min_slider.set(config.DEFAULT_F_MIN)

    # f_max Slider
    f_max_label = ttk.Label(root, text='f_max:')
    f_max_slider = tk.Scale(root, from_=8000, to=16000, resolution=100, orient='horizontal')
    f_max_slider.set(config.DEFAULT_F_MAX)

    # Power Dropdown (Combobox)
    power_label = ttk.Label(root, text='Power:')
    power_toggle = ttk.Combobox(root, values=[("Power", 2.0), ("Magnitude", 1.0)])
    power_toggle.set(config.DEFAULT_POWER)

    # Threshold Slider
    threshold_label = ttk.Label(root, text='Threshold:')
    threshold_slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient='horizontal')
    threshold_slider.set(config.DEFAULT_THRESHOLD)

    # Spike Plot Radio Button
    spike_plot_radio_label = ttk.Label(root, text='Figs:')
    spike_plot_radio = ttk.Combobox(root, values=['spikes', 'distribution'])
    spike_plot_radio.set('spikes')

    # Spike Duration Slider
    spike_period_slider_label = ttk.Label(root, text='range:')
    spike_periode_slider = tk.Scale(root, from_=0, to=mel_config.step_duration, resolution=10, orient='horizontal')
    spike_periode_slider.set(150)

    # Channel Slider
    channel_slider_label = ttk.Label(root, text='Ch:')
    channel_slider = tk.Scale(root, from_=1, to=mel_config.n_mels, resolution=1, orient='horizontal')
    channel_slider.set(1)

    # Frequency Calculation and Label
    def freq_calc():
        _, _, spikes, _, _ = FU.generate_spikes(audio_sample, mel_config, spikes_data.threshold)
        max_start, max_stop, max_count = FU.find_max_interval(spikes, channel_slider.get(), spike_periode_slider.get())
        mel_time_resolution = mel_config.time_resolution
        range_steps = max_stop - max_start
        occurance_freq = max_count / (mel_time_resolution * range_steps)
        return occurance_freq, mel_time_resolution, max_start, max_stop, int(max_count)

    spk_freq_label = ttk.Label(root, text="")

    def update_label():
        occurance_freq, mel_time_resolution, max_start, max_stop, max_count = freq_calc()
        spk_freq_label.config(
            text=f"ch:{channel_slider.get()} rate:{occurance_freq:.1f}[Hz]\n"
                 f"{max_start} _ {max_stop} cnt:{max_count}\n"
                 f"dt:{mel_time_resolution * 1000:.2f}[msec] 1/dt:{1 / (mel_time_resolution * 1000):.1f}kHz"
        )

    spike_periode_slider.config(command=lambda _: update_label())
    channel_slider.config(command=lambda _: update_label())
    hop_length_slider.config(command=lambda _: update_label())

    # Filter Choice Radio Button
    filter_choice_label = ttk.Label(root, text='filter used:')
    filter_choice_radio = ttk.Combobox(root, values=['standard', 'custom'])
    filter_choice_radio.set('standard')

    # Mel Plot Radio Button
    mel_plot_radio_label = ttk.Label(root, text='Figs:')
    mel_plot_radio = ttk.Combobox(root, values=['sptrgm', 'filter'])
    mel_plot_radio.set('sptrgm')

    # Arrange widgets in a grid layout for better UI
    widgets = [
        (directory_dropdown_label, directory_dropdown),
        (file_slider_label, file_slider),
        (n_fft_label, n_fft_input),
        (hop_length_label, hop_length_slider),
        (n_mels_label, n_mels_slider),
        (f_min_label, f_min_slider),
        (f_max_label, f_max_slider),
        (power_label, power_toggle),
        (threshold_label, threshold_slider),
        (spike_plot_radio_label, spike_plot_radio),
        (spike_period_slider_label, spike_periode_slider),
        (channel_slider_label, channel_slider),
        (spk_freq_label, None),
        (filter_choice_label, filter_choice_radio),
        (mel_plot_radio_label, mel_plot_radio)
    ]

    row = 0
    for label, widget in widgets:
        label.grid(row=row, column=0, padx=5, pady=5, sticky='w')
        if widget:
            widget.grid(row=row, column=1, padx=5, pady=5, sticky='w')
        row += 1

    return (
        directory_dropdown, file_slider, n_fft_input, hop_length_slider, 
        n_mels_slider, f_min_slider, f_max_slider, power_toggle, 
        threshold_slider, spike_plot_radio, spike_periode_slider, 
        spk_freq_label, channel_slider, filter_choice_radio, mel_plot_radio, spk_freq_label
    )
