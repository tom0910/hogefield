#widgets_setup.py

import ipywidgets as widgets
import config.config as config
import utils.functional as FU

# Function to set up widgets
def create_widgets(audio_sample, mel_config, spikes_data):
    print("create widget function executing ..")
    directory_dropdown = widgets.Dropdown(
        options=audio_sample.get_directories(),
        value=config.DEFAULT_DIRECTORY,
        description='Directory:',
        layout=widgets.Layout(width='200px')
    )

    file_slider = widgets.IntSlider(
        min=0,
        max=len(audio_sample.get_files()) - 1,
        step=1,
        value=0,
        description='File Index:',
        layout=widgets.Layout(width='300px')
    )

    n_fft_input = widgets.IntText(
        value=config.DEFAULT_N_FFT,
        description='n_fft:',
        layout=widgets.Layout(width='200px')
    )

    hop_length_slider = widgets.IntSlider(
        min=1, max=512, step=1, value=config.DEFAULT_HOP_LENGTH, description='Hop Length'
    )

    n_mels_slider = widgets.IntSlider(
        min=2, max=22, step=1, value=config.DEFAULT_N_MELS, description='n_mels'
    )

    f_min_slider = widgets.IntSlider(
        min=0, max=8000, step=100, value=config.DEFAULT_F_MIN, description='f_min'
    )

    f_max_slider = widgets.IntSlider(
        min=8000, max=16000, step=100, value=config.DEFAULT_F_MAX, description='f_max'
    )

    power_toggle = widgets.Dropdown(
        options=[("Power", 2.0), ("Magnitude", 1.0)],
        value=config.DEFAULT_POWER,
        description="Power:",
        layout=widgets.Layout(width='200px')
    )
    
    threshold_slider = widgets.FloatSlider(
        min=0, max=1, step=0.01, value=config.DEFAULT_THRESHOLD, description='Threshold', layout=widgets.Layout(width='300px')
    )
    
    spike_plot_radio = widgets.RadioButtons(
        options=['spikes', 'distribution'],
        description='Figs:'
    )

    # IntSlider for spike duration
    spike_periode_slider = widgets.IntSlider(
        min=0, max=mel_config.step_duration, step=10, value=150, 
        description='range:', layout=widgets.Layout(width='300px')
    )
    
    # IntSlider for channel number
    channel_slider = widgets.IntSlider(
        min=1, max=mel_config.n_mels, step=1, value=1,
        description='Ch:', layout=widgets.Layout(width='300px')
    )

    # Helper function to calculate frequency
    def freq_calc():
        _, _, spikes,_,_ = FU.generate_spikes(audio_sample, mel_config, spikes_data.threshold)
        max_start, max_stop, max_count = FU.find_max_interval(spikes, channel_slider.value, spike_periode_slider.value)
        mel_time_resolution = mel_config.time_resolution
        range_steps = max_stop - max_start
        occurance_freq = max_count / (mel_time_resolution * range_steps)
        return occurance_freq, mel_time_resolution, max_start, max_stop, int(max_count)
    
    # Label to display frequency
    spk_freq_label = widgets.HTML(
        value=f"<div style='text-align: center;'>"
        f"ch:{channel_slider.value} rate:{freq_calc()[0]:.1f}[Hz] <br>"
        f"{freq_calc()[2]} _ {freq_calc()[3]} cnt:{freq_calc()[4]} "
        f"dt:{freq_calc()[1] * 1000:.2f}[msec] 1/dt:{1/(freq_calc()[1] * 1000):.1f}kHz"
        f"</div>",
        layout=widgets.Layout(width="300px")
    )
    
    # Update the label based on changes in sliders
    def update_label(change):
        spk_freq_label.value = (
            f"<div style='text-align: center;'>"
            f"ch:{channel_slider.value} rate:{freq_calc()[0]:.1f}[Hz] <br>"
            f"{freq_calc()[2]} _ {freq_calc()[3]} cnt:{freq_calc()[4]} "
            f"dt:{freq_calc()[1] * 1000:.2f}[msec] 1/dt:{1/(freq_calc()[1] * 1000):.1f}kHz"
            f"</div>"
        )

    # Attach observers to both sliders
    spike_periode_slider.observe(update_label, names='value')
    channel_slider.observe(update_label, names='value')    
    hop_length_slider.observe(update_label, names='value')
    spikes_data.add_observer(update_label)
    
    # Display slider and label side by side using HBox
    spike_period_slider_combo = widgets.VBox([channel_slider, spike_periode_slider, spk_freq_label])  

    filter_choice_radio = widgets.RadioButtons(
        options=['standard', 'custom'],
        description='filter used:',
        value='standard'
    )

    mel_plot_radio = widgets.RadioButtons(
        options=['sptrgm', 'filter'],
        description='Figs:',
        value='sptrgm'
    )
    
    return (
        directory_dropdown, file_slider, n_fft_input, hop_length_slider, 
        n_mels_slider, f_min_slider, f_max_slider, power_toggle, 
        threshold_slider, spike_plot_radio, spike_periode_slider, 
        spike_period_slider_combo, spk_freq_label, channel_slider,
        filter_choice_radio, mel_plot_radio, spk_freq_label
    )

