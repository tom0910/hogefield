import numpy as np
import soundfile as sf
import tkinter as tk
from display.tk_widgets_setup import create_widgets, initialize_plots, create_plots_frame, create_plots_frame_simple
from core.AudioSample import AudioSample
from core.MelSpectrogramConfig import MelSpectrogramConfig
from core.Spikes import Spikes
from display.tk_display_functions import xy_waveform_calc, save_audio, mel_spctgrm_calc, mel_spctgrm_calc2
from display.plotting import plot_waveform, plot_mel_spectrogram, plot_spikes, plot_distribution, plot_mel_spectrogram_inv, plot_waveform_with_slider
from utils.widget_sync_utils import set_audio_sample_from_widget_values, set_mel_config_from_widget_values
import utils.functional as FU
import config.config as C
import torch

if __name__ == "__main__":

    # Initialize AudioSample and MelSpectrogramConfig
    BASE_PATH="/project/data/ZTWF"
    DEFAULT_DIRECTORY="sine"
    DEFAULT_FILE_INDEX=0
    audio_sample = AudioSample(BASE_PATH, DEFAULT_DIRECTORY, DEFAULT_FILE_INDEX)
    mel_config = MelSpectrogramConfig(
        audio_sample,
        n_fft=512,
        hop_length=20,
        n_mels=40,
        f_min=300,
        f_max=8000,
        power=1.0,
        filter_type="narrowband",
        toggle_mel_filter="spktgrm" # is used?
    )
    DEFAULT_THRESHOLD = 0.0000420 
    spikes_data = Spikes(threshold=DEFAULT_THRESHOLD)
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Spike Generation Sensory Lab")
    root.geometry("1600x800")
    # create plots
    widget_frame, out_frame_audio_wave_plt, out_frame_mel_sptgrm_plt, out_spike_raster_plt,  out_rev_spike_raster_plt, out_rev_mel_sptgrm_plt, empty, empty2 = create_plots_frame_simple(root)
    
    # initialize_plots 
    frames = [
            out_frame_audio_wave_plt,
            out_frame_mel_sptgrm_plt,
            out_spike_raster_plt,
            out_rev_spike_raster_plt,
            out_rev_mel_sptgrm_plt,
            empty,
            empty2,
        ]

    figs, axes, canvases = initialize_plots(frames)

    def update_plot():
        print("update plot triggered")
        #create plots
        set_audio_sample_from_widget_values(audio_sample, directory_dropdown, file_slider)
        #save audio to listen back original database wav file
        save_audio(audio_sample, file_name="original_waveform.wav", save_dir="src/run/SaveAudio")
        time, amplitude = xy_waveform_calc(audio_sample)
        plot_waveform(time, amplitude, canvases[0], axes[0],is_labels=None)
        time_plt = time[6400:8000]
        amplitude_plt = amplitude[6400:8000]
        print(f"amplitude: type={type(amplitude)}, shape={getattr(amplitude, 'shape', 'N/A')}")
        # plot_waveform(time_plt, amplitude_plt, canvases[1], axes[1], is_labels=None)
        plot_waveform_with_slider(time, amplitude, canvases[1], axes[1], is_label=None)
        print("update_melspctgrm() triggered")
        set_mel_config_from_widget_values(
            mel_config, 
            # n_fft_input, 
            n_fft_slider,
            hop_length_slider, n_mels_slider, f_min_slider, f_max_slider, 
            power_toggle, 
            filter_choice_widget,
            mel_filter_plot_radio_widget
        )
        mel_spectrogram, updated_audio_sample, updated_mel_config, sample_rate, custom_mel_scale = mel_spctgrm_calc2(audio_sample, mel_config)
        plot_mel_spectrogram(mel_spectrogram, updated_audio_sample, updated_mel_config, sample_rate, custom_mel_scale, canvases[2], axes[2], is_tick_color=None)
        
        update_plot_spike()
        
    def update_plot_spike():
        print("update spike plot tiggered")
        #objects updates width widgets' values
        spikes_data.threshold = float(threshold_slider.get())
        spike_threshold_updated = spikes_data.threshold
        plot_choice = spike_plot_radio_widget.get()
        
        #calculation:
        num_neurons, num_spike_index, spikes, original_min_max, _ = FU.generate_spikes2(audio_sample, mel_config, spike_threshold_updated, norm_inp=True, norm_cumsum=False)
        print("spikes calculated")
        # plot spike trains or distribution
        if plot_choice == C.DEFAULT_SPIKE_PLT_PICK:
            #plot:
            plot_spikes(spikes=spikes, num_neurons=num_neurons, num_spike_index=num_spike_index, canvas=canvases[3], ax=axes[3])
            print("spikes plotted")
        elif plot_choice == C.DEFAULT_DIST_PLT_PICK:
            #calculations:
            mel_spectrogram, _, _ = FU.get_mel_spectrogram(audio_sample, mel_config)
            mel_spectrogram_noramlized = FU.normalize(mel_spectrogram, normalize_to=1)
            #plot:
            plot_distribution(mel_spectrogram_noramlized, canvases[3], axes[3])
        else:
            raise ValueError("Invalid plot_choice value. Use 'spikes' or 'distribution'.")
        print("reconstructed plots tiggered")
        #calculation of melspectrogram reconstruction:
        mel_spectrogram_approx=FU.inverse_generate_spikes2(spikes, mel_config, audio_sample.sample_rate, spike_threshold_updated, norm_inp=True, norm_cumsum=False, orig_min_max=original_min_max)
        print("melspctgrm reconstructed")
        #plot:
        print("ori max:", original_min_max[1] )
        plot_mel_spectrogram_inv(
            mel_spectrogram=mel_spectrogram_approx,
            audio_sample=audio_sample,
            hop_length=mel_config.hop_length,
            sample_rate=audio_sample.sample_rate,
            custom_mel_scale=None,  # Or pass if custom_mel_scale is needed
            title="Approximated Mel Spectrogram",
            canvas=canvases[4],
            ax = axes[4],
            # original_max=original_min_max[1]
        ) 
        #calculation of audio waveform reconstruction:
        waveform_approx = FU.inverse_mel_to_waveform2(mel_spectrogram=mel_spectrogram_approx, mel_config=mel_config , sample_rate=audio_sample.sample_rate)
        # Convert waveform to numpy if it's a torch.Tensor
        if isinstance(waveform_approx, torch.Tensor):
            waveform_approx = waveform_approx.numpy()
        audio_length_in_sec = waveform_approx.size / mel_config.sample_rate
        print(f"wave approx in sec, wave size, srate: {audio_length_in_sec}, {waveform_approx.size}, {mel_config.sample_rate}")
        time = np.linspace(0, audio_length_in_sec, num=waveform_approx.size)
        print("audio waveform data reconstructed")
        #plot waveform
        plot_waveform_with_slider(time, waveform_approx, canvases[5], axes[5], is_label=None)
        # plot_waveform(time[6400:8000], waveform_approx[6400:8000], canvases[5], axes[5], is_labels=None)
        plot_waveform(time, waveform_approx, canvases[6], axes[6], is_labels=None)
        # save audio file to listen
        save_audio(waveform=waveform_approx,sample_rate=mel_config.sample_rate, file_name="reconstructed_waveform.wav", save_dir="src/run/SaveAudio")
        

    # Create Widgets using Tkinter
    (
        directory_dropdown, 
        file_slider, 
        file_slider_entry,
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
        filename_entry,
        save_matdata_widget      
    ) = create_widgets(audio_sample, mel_config, spikes_data, widget_frame)
    
    
    
    directory_dropdown.bind("<<ComboboxSelected>>", lambda e: update_plot())
    file_slider.config(command=lambda v: update_plot())
    file_slider_entry.bind('<Return>', lambda event: update_plot())
    
    n_fft_slider.config(command=lambda v: update_plot())
    n_fft_entry.bind('<Return>', lambda event: update_plot())
    
    # observ hop length entry
    hop_length_slider.config(command=lambda v: update_plot())
    hop_length_entry.bind('<Return>', lambda event: update_plot())
    # observ number of mels entry
    n_mels_slider.config(command=lambda v: update_plot())
    n_mels_entry.bind('<Return>',lambda event: update_plot())
    # observer max and min frequencies
    f_min_slider.config(command=lambda v: update_plot())
    f_max_slider.config(command=lambda v: update_plot())
    # setting betwen power and magnitude
    power_toggle.bind(update_plot)
    # observer that should change filter type in sptgrm
    filter_choice_widget.bind(update_plot)
    # observer to change canvas btwn spectrogram and filter type used 
    mel_filter_plot_radio_widget.bind(update_plot)

    # observ spike related chages:
    threshold_slider.config(command=lambda v: update_plot())
    threshold_entry.bind('<Return>', lambda event: update_plot())
    spike_plot_radio_widget.bind(update_plot)    
    
    #SET DEFAULT Widget Values:
    directory_dropdown.set("piano_scale")  # Default to "piano_scale" name
    n_fft_var = tk.IntVar(value=400)  # Create a new IntVar
    n_fft_entry.configure(textvariable=n_fft_var)  # Link the IntVar to the Entry
    n_fft_slider.configure(variable=n_fft_var)
    f_min_slider.set(0)
    f_max_slider.set(800)
    n_mels_slider.set(40)
    
    
    
    update_plot()
    update_plot_spike()
    # set_audio_sample_from_widget_values(audio_sample, directory_dropdown, file_slider)
    root.mainloop()

# import torchaudio
# waveform, sample_rate = torchaudio.load("composed_waveform.wav")
# import utils.preprocess_collate as PC
# mel_spectrogram, sample_rate, custom_mel_scale = PC.gen_melspecrogram_common(
#         waveform, 22, sample_rate, 300, 8000, 'narrowband', 512, 20, 1.0, center=True
#     )


