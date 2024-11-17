#plotting.py
import matplotlib.pyplot as plt
import numpy as np
import torch as torch
import ipywidgets as widgets # ideiglenes
import utils.functional as FU
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Function to update the audio waveform
def plot_waveform(x_values, waveform, canvas, ax):
    # Clear the existing plot content
    ax.clear()
    
    title="Audio Amplitude Over Time"
    # Plot the waveform
    ax.plot(x_values, waveform)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.figure.tight_layout()  # Minimize margin
    
    # Redraw the canvas with updated data
    canvas.draw()

def plot_mel_spectrogram(mel_spec, audio_sample, mel_config, sample_rate, custom_mel_scale, canvas, ax):
    # Ensure `_colorbar` exists as a default attribute
    if not hasattr(ax, "_colorbar"):
        ax._colorbar = None

    ax.clear()            
            
    mel_spec = mel_spec.squeeze().numpy()
    num_mel_bins = mel_spec.shape[0]

    yticks = np.arange(1, num_mel_bins + 1)
    time_axis = mel_spec.shape[-1]
    x_ticks = np.linspace(0, time_axis - 1, 4)
    x_ticks = np.round(x_ticks).astype(int)
    hop_length = mel_config.hop_length
    x_tick_labels = [f'{(tick * hop_length) / sample_rate:.2f}' for tick in x_ticks]

    total_samples = int(time_axis)

    # Show the colorbar
    toggle_colorbar_visibility(ax, visible=True)
    
    # Update or plot the spectrogram without clearing the axes
    if ax.images:
        # Update the existing image
        ax.images[0].set_data(mel_spec)
    else:
        # Add a new image
        img = ax.imshow(mel_spec, aspect="auto", origin="lower", vmin=None, vmax=None)

    if mel_config.toggle_mel_filter == "sptrgm":
        img = ax.imshow(mel_spec, aspect="auto", origin="lower", vmin=None, vmax=None)
        ax.set_title(f"Mel Spectrogram - {audio_sample.selected_directory}", fontsize=12)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_xlabel("Time [sec]", fontsize=10)
        ax.set_yticks(yticks - 1)
        ax.set_yticklabels(yticks)
        ax.set_ylabel("Mel Bins", fontsize=10)

        # Handle the colorbar
        if ax._colorbar:
            # Update the colorbar's data
            ax._colorbar.update_normal(img)
        else:
            # Add a new colorbar
            cbar = ax.figure.colorbar(img, ax=ax, orientation="vertical", format="%0.1f")
            cbar.set_ticks(np.arange(0, 1.1, 0.1))
            ax._colorbar = cbar

        ax.annotate(
            f"Total Samples: {int(time_axis)}",
            xy=(0.5, -0.2),
            xycoords="axes fraction",
            ha="center",
            fontsize=12,
        )
        for ytick in yticks - 1:
            ax.axhline(y=ytick + 0.5, color="orange", linestyle=":", linewidth=1.5)
        
        # Redraw the canvas
        # ax.figure.tight_layout()
        canvas.draw()
        
    elif mel_config.toggle_mel_filter == "filter":
        # print(f"Before removal: {ax._colorbar}")
        toggle_colorbar_visibility(ax, visible=False)
        # if not hasattr(ax, "_colorbar"):
        #     ax._colorbar.ax.remove()
        #     ax._colorbar = None
            
        # Safely remove the colorbar
        # safely_remove_colorbar(ax)
        # hide_colorbar(ax)            
        
        # Clear the current Axes
        # print(f"After removal: {ax._colorbar}")
        ax.clear()

        # Plot the filter banks
        plot_filterbanks(
            custom_mel_scale.fb,
            custom_mel_scale.all_freqs,
            custom_mel_scale.f_pts,
            custom_mel_scale.spread,
            canvas, ax
        )
        # Redraw the canvas
        canvas.draw()
        
def toggle_colorbar_visibility(ax, visible=True):
    """
    Toggle the visibility of the colorbar on a given Axes.

    Args:
        ax (matplotlib.axes.Axes): The Axes containing the colorbar.
        visible (bool): Whether to show or hide the colorbar.
    """
    if hasattr(ax, "_colorbar") and ax._colorbar:
        try:
            cbar = ax._colorbar
            # Toggle visibility of colorbar
            cbar.ax.set_visible(visible)
            
            # Optionally, clear the colorbar outline and ticks when hiding
            if not visible:
                cbar.outline.set_visible(False)
                cbar.ax.tick_params(labelleft=False, labelright=False)
            else:
                cbar.outline.set_visible(True)
                cbar.ax.tick_params(labelleft=True, labelright=False)
        except Exception as e:
            print(f"Error while toggling colorbar visibility: {e}")

# def safely_remove_colorbar(ax):
#     """
#     Safely removes the colorbar from the given Axes.

#     Args:
#         ax (matplotlib.axes.Axes): The Axes object from which to remove the colorbar.
#         canvas (FigureCanvasTkAgg): The canvas tied to the figure.
#     """
#     if hasattr(ax, "_colorbar") and ax._colorbar:
#         print(f"Before removal: {ax._colorbar}, Colorbar axes: {ax._colorbar.ax}")

#         try:
#             if ax._colorbar.ax:  # Ensure the colorbar's Axes is valid
#                 ax._colorbar.ax.remove()
#         except AttributeError as e:
#             print(f"Error while removing colorbar: {e}")
#         finally:
#             ax._colorbar = None  # Clear the colorbar reference
        
#         print(f"After removal: {ax._colorbar}")  # Confirm removal success
        
        

def plot_filterbanks(filter_banks, all_freqs, f_pts, spread, canvas, ax):
    """
    Plot the filter banks on a given Matplotlib canvas and axes.

    Args:
        filter_banks (torch.Tensor): The filter bank tensor of shape (n_freqs, n_filters).
        all_freqs (torch.Tensor): Frequencies corresponding to the rows in filter_banks.
        f_pts (list or torch.Tensor): Center frequencies of the filters.
        spread (int): Spread of each filter.
        canvas (FigureCanvasTkAgg): The Matplotlib canvas for rendering the plot.
        ax (matplotlib.axes.Axes): The axes to plot on.
    """
    # Clear the axes
    ax.clear()

    # Plot each filter bank
    for i in range(filter_banks.shape[1]):
        ax.plot(
            all_freqs.numpy(),
            filter_banks[:, i].numpy(),
            label=f"{f_pts[i]:.0f} Hz"
        )

    # Set axis labels, title, and other details
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Triangular Filter Bank with Spread of ±{spread} Hz")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize='small')
    ax.grid(True)

    # Trigger canvas redraw
    # canvas.draw()



# def plot_filterbanks(filter_banks, all_freqs, f_pts, spread):
#     """
#     Plot the filter banks.

#     Args:
#         filter_banks (torch.Tensor): The filter bank tensor of shape (n_freqs, n_filters).
#         all_freqs (torch.Tensor): Frequencies corresponding to the rows in filter_banks.
#         f_pts (list or torch.Tensor): Center frequencies of the filters.
#         spread (int): Spread of each filter.
#     """
#     # plt.figure(figsize=(12, 6))
#     for i in range(filter_banks.shape[1]):
#         plt.plot(all_freqs.numpy(), filter_banks[:, i].numpy(), label=f"{f_pts[i]:.0f} Hz")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Amplitude")
#     plt.title(f"Triangular Filter Bank with Spread of ±{spread} Hz")
#     plt.legend()
#     plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize='small')
#     plt.grid(True)
#     # plt.show()

def plot_spikes(audio_sample, mel_config, spikes_data, plot_radio, output_frame):

    # Clear the output_frame
    for widget in output_frame.winfo_children():
        widget.destroy()

    # Create a Matplotlib figure and axis        
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

    # Embed the Matplotlib figure in the Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=output_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)        

# def plot_mel_spectrogram_inv(
#     mel_spectrogram,
#     audio_sample,
#     mel_config,
#     sample_rate,
#     custom_mel_scale,
#     output_widget,
#     title="Mel Spectrogram Approximation"
# ):
#     """
#     Plot a mel spectrogram approximation in a specified widget.

#     Args:
#         mel_spectrogram (torch.Tensor or np.ndarray): The mel spectrogram data to plot.
#         audio_sample: Audio sample information, used to display directory in title.
#         mel_config: Configuration object with settings like number of mel bins.
#         sample_rate (int): Sample rate of the audio.
#         custom_mel_scale: Custom mel scale for additional filter information if needed.
#         output_widget: The widget to display the plot.
#         title (str): Title for the plot (default is "Mel Spectrogram Approximation").
#     """
#     with output_widget:
#         output_widget.clear_output(wait=True)
        
#         # Convert to numpy if it's a torch.Tensor
#         if isinstance(mel_spectrogram, torch.Tensor):
#             mel_spectrogram = mel_spectrogram.squeeze().numpy()
        
#         fig, ax = plt.subplots(figsize=(6, 4))
#         img = ax.imshow(mel_spectrogram, aspect='auto', origin='lower', vmin=None, vmax=None)
        
#         # Configure plot title with directory information
#         ax.set_title(f'{title} - {audio_sample.selected_directory}', fontsize=12)
        
#         # Configure x-ticks for time in seconds
#         time_axis = mel_spectrogram.shape[-1]
#         hop_length = mel_config.hop_length
#         x_ticks = np.linspace(0, time_axis - 1, 4).astype(int)
#         x_tick_labels = [f'{(tick * hop_length) / sample_rate:.2f}' for tick in x_ticks]
        
#         ax.set_xticks(x_ticks)
#         ax.set_xticklabels(x_tick_labels)
#         ax.set_xlabel('Time [sec]', fontsize=10)
        
#         # Configure y-ticks for Mel bins
#         num_mel_bins = mel_spectrogram.shape[0]
#         yticks = np.arange(1, num_mel_bins + 1)
#         ax.set_yticks(yticks - 1)  # Adjust to match positions with Mel bins
#         ax.set_yticklabels(yticks)
#         ax.set_ylabel('Mel Bins', fontsize=10)
        
#         # Add colorbar with finer ticks
#         cbar = plt.colorbar(img, ax=ax, orientation='vertical', format="%0.1f")
#         cbar.set_ticks(np.linspace(mel_spectrogram.min(), mel_spectrogram.max(), 10))
        
#         plt.show()

def plot_mel_spectrogram_inv(
    mel_spectrogram,
    audio_sample,
    mel_config,
    sample_rate,
    custom_mel_scale,
    output_frame,
    title="Mel Spectrogram Approximation"
):
    """
    Plot a mel spectrogram approximation in a Tkinter frame.

    Args:
        mel_spectrogram (torch.Tensor or np.ndarray): The mel spectrogram data to plot.
        audio_sample: Audio sample information, used to display directory in title.
        mel_config: Configuration object with settings like number of mel bins.
        sample_rate (int): Sample rate of the audio.
        custom_mel_scale: Custom mel scale for additional filter information if needed.
        output_frame: The Tkinter frame to embed the plot.
        title (str): Title for the plot (default is "Mel Spectrogram Approximation").
    """
    # Clear previous content in the frame
    for widget in output_frame.winfo_children():
        widget.destroy()

    # Convert to numpy if it's a torch.Tensor
    if isinstance(mel_spectrogram, torch.Tensor):
        mel_spectrogram = mel_spectrogram.squeeze().numpy()

    # Create the Matplotlib figure and axis
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

    # Embed the Matplotlib figure into the Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=output_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)



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