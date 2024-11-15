def set_audio_sample_from_widget_values(audio_sample, directory_dropdown, file_slider):
    """Set `audio_sample` properties based on widget values."""
    audio_sample.selected_directory = directory_dropdown.get()
    audio_sample.file_index = int(file_slider.get())

def set_mel_config_from_widget_values(mel_config, n_fft_input, hop_length_slider, n_mels_slider, f_min_slider, f_max_slider, power_toggle, filter_choice_radio , mel_plot_radio):
    """Set `mel_config` properties based on widget values."""
    # mel_config.refresh_audio_sample_rate()
    # mel_config.n_fft = int(n_fft_input.get())
    # mel_config.hop_length = int(hop_length_slider.get())
    mel_config.update_from_widgets(
        n_fft=int(n_fft_input.get()),
        hop_length=int(hop_length_slider.get()),
        n_mels=int(n_mels_slider.get()),
        f_min=int(f_min_slider.get()),
        f_max=int(f_max_slider.get()),
        power=power_toggle.get(),
        filter_type=filter_choice_radio.get(),
        toggle_mel_filter=mel_plot_radio.get()
    )
    
