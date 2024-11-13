import matplotlib.pyplot as plt
import numpy as np
import torch as torch
import ipywidgets as widgets # ideiglenes
import utils.functional as FU

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_waveform(x_values, waveform, title="Audio Amplitude Over Time", ax=None, output_frame=None):
    """
    Plot the waveform using a given axis or create a new figure.

    Args:
        x_values (array-like): The time values for the waveform.
        waveform (array-like): The waveform values.
        title (str, optional): The title for the plot. Default is "Audio Amplitude Over Time".
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure is created.
        output_frame (tk.Frame, optional): Tkinter frame to embed the plot. If None, plt.show() is called.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Plotting on the provided axis
    ax.plot(x_values, waveform)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.figure.tight_layout()  # Minimize margin

    if output_frame:
        # If an output_frame is provided, embed the plot into the Tkinter frame.
        for widget in output_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(ax.figure, master=output_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    else:
        # Show the plot if no output_frame is specified
        plt.show()



# def plot_waveform(x_values, waveform, title="Audio Amplitude Over Time"):
#     plt.figure(figsize=(6, 4))
#     plt.plot(x_values, waveform)
#     plt.title(title)
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
#     plt.tight_layout() #minimize margin
#     plt.show()


import matplotlib.pyplot as plt
import torch

def plot_filterbanks(filter_banks, all_freqs, f_pts, spread):
    """
    Plot the filter banks.

    Args:
        filter_banks (torch.Tensor): The filter bank tensor of shape (n_freqs, n_filters).
        all_freqs (torch.Tensor): Frequencies corresponding to the rows in filter_banks.
        f_pts (list or torch.Tensor): Center frequencies of the filters.
        spread (int): Spread of each filter.
    """
    # plt.figure(figsize=(12, 6))
    for i in range(filter_banks.shape[1]):
        plt.plot(all_freqs.numpy(), filter_banks[:, i].numpy(), label=f"{f_pts[i]:.0f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"Triangular Filter Bank with Spread of ±{spread} Hz")
    plt.legend()
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize='small')
    plt.grid(True)
    # plt.show()

def plot_mel_spectrogram(mel_spec, audio_sample, mel_config, sample_rate, custom_mel_scale,output_widget):
    mel_spec = mel_spec.squeeze().numpy()  # Convert to numpy for easier manipulation
    num_mel_bins = mel_spec.shape[0]
    
    # Y-axis ticks for Mel bins
    yticks = np.arange(1, num_mel_bins + 1)
    
    # X-axis ticks for time
    time_axis = mel_spec.shape[-1]
    x_ticks = np.linspace(0, time_axis - 1, 4)
    x_ticks = np.round(x_ticks).astype(int)
    hop_length = mel_config.hop_length  # Retrieve hop_length from mel_config
    x_tick_labels = [f'{(tick * hop_length) / sample_rate:.2f}' for tick in x_ticks]

    total_samples = int(time_axis)
    
    # Plot the mel spectrogram in the specified widget
    with output_widget:
        output_widget.clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        if mel_config.toggle_mel_filter == "sptrgm":
            img = ax.imshow(mel_spec, aspect='auto', origin='lower', vmin=None, vmax=None)
            ax.set_title(f'Mel Spectrogram - {audio_sample.selected_directory}', fontsize=12)
            
            # Set x-ticks for time in seconds
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel('Time [sec]', fontsize=10)
            
            # Set y-ticks for Mel bins
            ax.set_yticks(yticks - 1)  # Adjust to match positions with Mel bins
            ax.set_yticklabels(yticks)
            ax.set_ylabel('Mel Bins', fontsize=10)
    
            # Add colorbar with finer ticks
            cbar = plt.colorbar(img, ax=ax, orientation='vertical', format="%0.1f")  # Show one decimal place
            cbar.set_ticks(np.arange(0, 1.1, 0.1))  # Ticks at intervals of 0.1 from 0 to 1
            
            
            # Adjust the annotation position to avoid overlap
            ax.annotate(
                f'Total Samples: {total_samples}', 
                xy=(0.5, -0.2),  # Move it lower by setting a larger negative y-coordinate
                xycoords='axes fraction', 
                ha='center', 
                fontsize=12
            )
            
            # Add orange horizontal dotted lines between Mel bins
            for ytick in yticks - 1:
                ax.axhline(y=ytick + 0.5, color='orange', linestyle=':', linewidth=1.5)

        elif mel_config.toggle_mel_filter == "filter":
            # Plot the filter banks
            plot_filterbanks(
                custom_mel_scale.fb,
                custom_mel_scale.all_freqs,
                custom_mel_scale.f_pts,
                custom_mel_scale.spread
            )
            
        # plt.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

def plot_spikes(audio_sample, mel_config, spikes_data, plot_radio, output_widget):

    with output_widget:
        output_widget.clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        if plot_radio == "distribution":
            
            mel_spectrogram, _, _ = FU.get_mel_spectrogram(audio_sample, mel_config)
            plt.hist(mel_spectrogram.flatten().numpy(), bins=100)
            plt.title(f'Data Distribution After Normalization {plot_radio}')
            plt.xlabel("Value")
            plt.ylabel("Data Count")

        elif plot_radio=="spikes":
            threshold = spikes_data.threshold
            num_neurons, num_spike_index, spikes,_,_ = FU.generate_spikes(audio_sample, mel_config, threshold, norm_inp=True, norm_cumsum=True)
            y_coords, x_coords = torch.nonzero(spikes, as_tuple=True)

            ax.scatter(x_coords, y_coords, s=1, c="black", label="")

            #calculate avarage rate of spiking in the highest count channel
            channel, count = find_max_one_channels(spikes)

            ax.annotate(
                f'channel number {channel} has a max avrg spikes of  {(count/spikes.shape[1]):.3f}', 
                xy=(0.5, -0.2),  # Move it lower by setting a larger negative y-coordinate
                xycoords='axes fraction', 
                ha='center', 
                fontsize=9
            )
            
            # Set up custom y-ticks and labels
            yticks = np.arange(0, num_neurons)
            ytick_labels = [f'neuron {i+1}' for i in yticks]  # Custom labels for each channel
            
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels)
            ax.set_ylim(-0.5, num_neurons - 0.5)  # Ensures all y-ticks are visible
            
            # Set limits and labels
            x_max_index = num_spike_index
            ax.set_xlim(0, x_max_index)
            ax.set_xlabel('Spike Event Index', fontsize=10)
            ax.set_ylabel('Channels', fontsize=10)
            ax.set_title("Positive Spikes Visualization", fontsize=12)
            # plt.show()

        plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_mel_spectrogram_inv(
    mel_spectrogram,
    audio_sample,
    mel_config,
    sample_rate,
    custom_mel_scale,
    output_widget,
    title="Mel Spectrogram Approximation"
):
    """
    Plot a mel spectrogram approximation in a specified widget.

    Args:
        mel_spectrogram (torch.Tensor or np.ndarray): The mel spectrogram data to plot.
        audio_sample: Audio sample information, used to display directory in title.
        mel_config: Configuration object with settings like number of mel bins.
        sample_rate (int): Sample rate of the audio.
        custom_mel_scale: Custom mel scale for additional filter information if needed.
        output_widget: The widget to display the plot.
        title (str): Title for the plot (default is "Mel Spectrogram Approximation").
    """
    with output_widget:
        output_widget.clear_output(wait=True)
        
        # Convert to numpy if it's a torch.Tensor
        if isinstance(mel_spectrogram, torch.Tensor):
            mel_spectrogram = mel_spectrogram.squeeze().numpy()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        img = ax.imshow(mel_spectrogram, aspect='auto', origin='lower', vmin=None, vmax=None)
        
        # Configure plot title with directory information
        ax.set_title(f'{title} - {audio_sample.selected_directory}', fontsize=12)
        
        # Configure x-ticks for time in seconds
        time_axis = mel_spectrogram.shape[-1]
        hop_length = mel_config.hop_length
        x_ticks = np.linspace(0, time_axis - 1, 4).astype(int)
        x_tick_labels = [f'{(tick * hop_length) / sample_rate:.2f}' for tick in x_ticks]
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_xlabel('Time [sec]', fontsize=10)
        
        # Configure y-ticks for Mel bins
        num_mel_bins = mel_spectrogram.shape[0]
        yticks = np.arange(1, num_mel_bins + 1)
        ax.set_yticks(yticks - 1)  # Adjust to match positions with Mel bins
        ax.set_yticklabels(yticks)
        ax.set_ylabel('Mel Bins', fontsize=10)
        
        # Add colorbar with finer ticks
        cbar = plt.colorbar(img, ax=ax, orientation='vertical', format="%0.1f")
        cbar.set_ticks(np.linspace(mel_spectrogram.min(), mel_spectrogram.max(), 10))
        
        plt.show()


def find_max_one_channels(spikes):
    # Count 1s on channels
    ones_count = torch.sum(spikes == 1, dim=1)
    # Look for max counts
    max_count = torch.max(ones_count)
    # channels where are maximum # of 1's
    max_channels = torch.where(ones_count == max_count)[0]
    max_channels = max_channels + 1 #now channel  0 is channel 1
    return max_channels.tolist(), max_count.item()

def find_max_interval(spikes, channel, time_step):
    """
    Finds the interval in the specified channel with the maximum number of 1's.
    
    Args:
        spikes (torch.Tensor): Tensor with shape (num_channels, num_time_steps) containing binary values.
        channel (int): The channel number where to perform the search.
        time_step (int): The size of the interval to consider.

    Returns:
        tuple: (start_index, end_index, max_count) where:
            - start_index is the starting time step of the interval with max 1's.
            - end_index is the end time step of this interval.
            - max_count is the count of 1's in this interval.
    """
    # Select the specified channel
    channel_data = spikes[channel]
    
    max_count = 0
    max_interval = (0, time_step)  # Initializing with the first interval
    
    # Slide the window across the time steps
    for start in range(len(channel_data) - time_step + 1):
        end = start + time_step
        current_count = channel_data[start:end].sum().item()  # Sum 1's in the current window

        if current_count > max_count:
            max_count = current_count
            max_interval = (start, end)

    return max_interval[0], max_interval[1], max_count