def set_audio_sample_from_widget_values(audio_sample, directory_dropdown, file_slider):
    """Set `audio_sample` properties based on widget values."""
    audio_sample.selected_directory = directory_dropdown.value
    audio_sample.file_index = file_slider.value

def set_mel_config_from_widget_values(mel_config, n_fft_input, hop_length_slider, n_mels_slider, f_min_slider, f_max_slider, power_toggle, filter_choice_radio , mel_plot_radio):
    """Set `mel_config` properties based on widget values."""
    # mel_config.refresh_audio_sample_rate()
    mel_config.update_from_widgets(
        n_fft=n_fft_input.value,
        hop_length=hop_length_slider.value,
        n_mels=n_mels_slider.value,
        f_min=f_min_slider.value,
        f_max=f_max_slider.value,
        power=power_toggle.value,
        filter_type=filter_choice_radio.value,
        toggle_mel_filter=mel_plot_radio.value
    )
    
