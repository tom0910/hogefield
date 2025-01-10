#plotting.py
import matplotlib.pyplot as plt
import numpy as np
import torch as torch
import ipywidgets as widgets # ideiglenes
import utils.functional as FU
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# import src.config.config as config
import config.config as config

# Function to update the audio waveform
def plot_waveform(x_values, waveform, canvas, ax, is_labels=True):
    # Clear the existing plot content
    ax.clear()
    title="Audio Amplitude Over Time"
    # Plot the waveform
    ax.plot(x_values, waveform)
    if is_labels:
        ax.set_title(title)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
    ax.figure.tight_layout()  # Minimize margin
    
    # Redraw the canvas with updated data
    canvas.draw()

from matplotlib.widgets import Slider  

def plot_waveform_with_slider(x_values, waveform, canvas, ax, is_label=True):
    # Clear the existing plot content
    ax.clear()

    # Initial range for plotting
    start_idx, end_idx = 3200, 4800

    # Plot the waveform
    line, = ax.plot(x_values[start_idx:end_idx], waveform[start_idx:end_idx])
    if is_label:
        ax.set_title("Audio Amplitude Over Time")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")

    if not hasattr(ax, "_slider"):
        # Create slider if it doesn't exist
        slider_ax = ax.figure.add_axes([0.2, 0.01, 0.65, 0.03])
        ax._slider = Slider(slider_ax, "Range", 0, len(x_values) - 1000, valinit=start_idx, valstep=1)
    else:
        # Update existing slider's range and reset its position
        ax._slider.valmin = 0
        ax._slider.valmax = len(x_values) - 1000
        ax._slider.set_val(start_idx)
        ax._slider.ax.set_visible(True)  # Make sure it's visible if previously hidden
    slider = ax._slider


    # Update the plot when slider value changes
    def update(val):
        start_idx = int(slider.val)
        end_idx = start_idx + 1000  # Fixed width of 1000 samples
        line.set_xdata(x_values[start_idx:end_idx])
        line.set_ydata(waveform[start_idx:end_idx])
        ax.relim()
        ax.autoscale_view()
        canvas.draw_idle()  # Redraw dynamically


    slider.on_changed(update)

    # Redraw the canvas with updated data
    canvas.draw()



# def plot_waveform_with_slider(x_values, waveform, canvas, ax):
#     # Clear the existing plot content
#     ax.clear()

#     # Initial range for plotting
#     start_idx, end_idx = 3200, 4800

#     # Plot the waveform
#     line, = ax.plot(x_values[start_idx:end_idx], waveform[start_idx:end_idx])
#     ax.set_title("Audio Amplitude Over Time")
#     ax.set_xlabel("Time (seconds)")
#     ax.set_ylabel("Amplitude")

#     # Add a slider
#     slider_ax = ax.figure.add_axes([0.2, 0.01, 0.65, 0.03])  # Position for the slider
#     slider = Slider(slider_ax, 'Range', 0, len(x_values) - 1000, valinit=start_idx, valstep=1)

#     # Update the plot when slider value changes
#     def update(val):
#         start_idx = int(slider.val)
#         end_idx = start_idx + 1000  # Fixed width of 1000 samples
#         line.set_xdata(x_values[start_idx:end_idx])
#         line.set_ydata(waveform[start_idx:end_idx])
#         ax.relim()
#         ax.autoscale_view()
#         canvas.draw_idle()

#     slider.on_changed(update)

#     # Redraw the canvas with updated data
#     canvas.draw()    

def plot_mel_spectrogram(mel_spec, audio_sample, mel_config, sample_rate, custom_mel_scale, canvas, ax, is_tick_color=True):
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
        if is_tick_color:
            for ytick in yticks - 1:
                ax.axhline(y=ytick + 0.5, color="orange", linestyle=":", linewidth=1.5)
        
        # Redraw the canvas
        # ax.figure.tight_layout()
        canvas.draw()
        
    elif mel_config.toggle_mel_filter == "filter":
        # print(f"Before removal: {ax._colorbar}")
        toggle_colorbar_visibility(ax, visible=False)

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

# def plot_spikes(audio_sample, mel_config, spikes_data, canvas, ax):
def plot_spikes(spikes,num_neurons, num_spike_index, canvas, ax):
    """
    Plot spikes on the given canvas and axes.

    Args:
        audio_sample (AudioSample): The audio sample data.
        mel_config (MelSpectrogramConfig): Mel spectrogram configuration.
        spikes_data (Spikes): Spikes data, including threshold.
        canvas (FigureCanvasTkAgg): The canvas to render the plot.
        ax (matplotlib.axes.Axes): The axes to render the plot.
    """
    # Clear previous plots
    ax.clear()

    # Retrieve threshold and calculate spikes
    # threshold = spikes_data.threshold
    # threshold = spike_threshold
    # num_neurons, num_spike_index, spikes, _, _ = FU.generate_spikes(
    #     audio_sample, mel_config, threshold, norm_inp=True, norm_cumsum=True
    # )

    # Get spike coordinates
    y_coords, x_coords = torch.nonzero(spikes, as_tuple=True)
    ax.scatter(x_coords, y_coords, s=1, c="black", label="")
    print(y_coords.shape, x_coords.shape)
    

    # Find the channel with the highest average spike rate
    channel, count = find_max_one_channels(spikes)

    # Annotate the plot with the highest average spike information
    ax.annotate(
        f"Channel number {channel} has a max avg spikes of {(count / spikes.shape[1]):.3f}",
        xy=(0.5, -0.2),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
    )

    # Set custom y-ticks and labels
    yticks = np.arange(0, num_neurons)
    ytick_labels = [f"Neuron {i + 1}" for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylim(-0.5, num_neurons - 0.5)

    # Set plot limits and labels
    x_max_index = num_spike_index
    ax.set_xlim(0, x_max_index)
    ax.set_xlabel("Spike Event Index", fontsize=10)
    ax.set_ylabel("Channels", fontsize=10)
    ax.set_title("Positive Spikes Visualization", fontsize=12)

    # Redraw the canvas
    canvas.draw()


# def plot_distribution(audio_sample, mel_config, canvas, ax):
def plot_distribution(mel_spectrogram, canvas, ax):    
    """
    Plots the histogram distribution of the Mel spectrogram values.

    Args:
        ax (matplotlib.axes.Axes): The Axes to draw the histogram.
        mel_spectrogram (torch.Tensor): The Mel spectrogram data.
        canvas (FigureCanvasTkAgg): The canvas where the plot will be displayed.
    """
    ax.clear()  # Clear Axes for reuse
   
    # mel_spectrogram, _, _ = FU.get_mel_spectrogram(audio_sample, mel_config)
    # print(f"Mel spectrogram shape: {mel_spectrogram.shape}")
    # print(f"Mel spectrogram max value: {mel_spectrogram.max()}")
    # print(f"Mel spectrogram min value: {mel_spectrogram.min()}")

    ax.hist(mel_spectrogram.flatten().numpy(), bins=500, color="blue", alpha=0.7)
    ax.set_title("Data Distribution After Normalization", fontsize=12)
    ax.set_xlabel("Value", fontsize=10)
    ax.set_ylabel("Data Count", fontsize=10)
    ax.grid(True)
    
    
    # Automatically adjust the y-axis
    # ax.relim()  # Recompute data limits
    # ax.autoscale()  # Automatically scale y-axis
    # Explicit y-lim a jobb vizuális érthetőségért
        # Ha automatikus skálázás nem működik jól
    max_y = ax.get_ylim()[1]  # Az aktuális automatikus felső határ
    ax.set_ylim(0, max_y*0.2)  # Fix skála beállítása

    # Redraw the canvas
    canvas.draw()


def plot_mel_spectrogram_inv(
    mel_spectrogram,
    audio_sample,
    # mel_config,
    hop_length,
    sample_rate,
    custom_mel_scale,
    title,
    canvas,
    ax,
    original_max=None, 
):
    """
    Plot a mel spectrogram approximation on a given Matplotlib Axes embedded in a Tkinter canvas.

    Args:
        mel_spectrogram (torch.Tensor or np.ndarray): The mel spectrogram data to plot.
        audio_sample: Audio sample information, used to display directory in title.
        mel_config: Configuration object with settings like number of mel bins.
        sample_rate (int): Sample rate of the audio.
        custom_mel_scale: Custom mel scale for additional filter information if needed.
        canvas (FigureCanvasTkAgg): The canvas where the plot will be displayed.
        ax (matplotlib.axes.Axes): The axes to render the plot.
        title (str): Title for the plot (default is "Mel Spectrogram Approximation").
    """
    # Clear the axes for reuse
    ax.clear()

    # Convert to numpy if it's a torch.Tensor
    if isinstance(mel_spectrogram, torch.Tensor):
        mel_spectrogram = mel_spectrogram.squeeze().numpy()

    # Plot the Mel spectrogram approximation
    # img = ax.imshow(mel_spectrogram, aspect='auto', origin='lower', vmin=None, vmax=None)
    
    vmax_value = original_max if original_max is not None else mel_spectrogram.max()
    img = ax.imshow(mel_spectrogram, aspect='auto', origin='lower', vmin=0, vmax=vmax_value)


    # Configure plot title with directory information
    ax.set_title(f'{title} - {audio_sample.selected_directory}', fontsize=12)

    # Configure x-ticks for time in seconds
    time_axis = mel_spectrogram.shape[-1]
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

    # Add or update the colorbar
    if not hasattr(ax, "_colorbar") or ax._colorbar is None:
        cbar = ax.figure.colorbar(img, ax=ax, orientation='vertical', format="%0.1f")
        # cbar.set_ticks(np.linspace(mel_spectrogram.min(), mel_spectrogram.max(), 10))
        cbar.set_ticks(np.linspace(mel_spectrogram.min(), vmax_value, 10))

        ax._colorbar = cbar
    else:
        ax._colorbar.update_normal(img)

    # Redraw the canvas
    canvas.draw()


# def plot_mel_spectrogram_inv(
#     mel_spectrogram,
#     audio_sample,
#     mel_config,
#     sample_rate,
#     custom_mel_scale,
#     title,
#     canvas,
#     ax,
# ):
#     """
#     Plot a mel spectrogram approximation on a given Matplotlib Axes embedded in a Tkinter canvas.

#     Args:
#         mel_spectrogram (torch.Tensor or np.ndarray): The mel spectrogram data to plot.
#         audio_sample: Audio sample information, used to display directory in title.
#         mel_config: Configuration object with settings like number of mel bins.
#         sample_rate (int): Sample rate of the audio.
#         custom_mel_scale: Custom mel scale for additional filter information if needed.
#         canvas (FigureCanvasTkAgg): The canvas where the plot will be displayed.
#         ax (matplotlib.axes.Axes): The axes to render the plot.
#         title (str): Title for the plot (default is "Mel Spectrogram Approximation").
#     """
#     # Clear the axes for reuse
#     ax.clear()

#     # Convert to numpy if it's a torch.Tensor
#     if isinstance(mel_spectrogram, torch.Tensor):
#         mel_spectrogram = mel_spectrogram.squeeze().numpy()

#     # Plot the Mel spectrogram approximation
#     img = ax.imshow(mel_spectrogram, aspect='auto', origin='lower', vmin=None, vmax=None)

#     # Configure plot title with directory information
#     ax.set_title(f'{title} - {audio_sample.selected_directory}', fontsize=12)

#     # Configure x-ticks for time in seconds
#     time_axis = mel_spectrogram.shape[-1]
#     hop_length = mel_config.hop_length
#     x_ticks = np.linspace(0, time_axis - 1, 4).astype(int)
#     x_tick_labels = [f'{(tick * hop_length) / sample_rate:.2f}' for tick in x_ticks]

#     ax.set_xticks(x_ticks)
#     ax.set_xticklabels(x_tick_labels)
#     ax.set_xlabel('Time [sec]', fontsize=10)

#     # Configure y-ticks for Mel bins
#     num_mel_bins = mel_spectrogram.shape[0]
#     yticks = np.arange(1, num_mel_bins + 1)
#     ax.set_yticks(yticks - 1)  # Adjust to match positions with Mel bins
#     ax.set_yticklabels(yticks)
#     ax.set_ylabel('Mel Bins', fontsize=10)

#     # Add or update the colorbar
#     if not hasattr(ax, "_colorbar") or ax._colorbar is None:
#         cbar = ax.figure.colorbar(img, ax=ax, orientation='vertical', format="%0.1f")
#         cbar.set_ticks(np.linspace(mel_spectrogram.min(), mel_spectrogram.max(), 10))
#         ax._colorbar = cbar
#     else:
#         ax._colorbar.update_normal(img)

#     # Redraw the canvas
#     canvas.draw()



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