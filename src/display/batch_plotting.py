import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_spikes(spikes, num_neurons, num_spike_index, canvas, ax):
    """
    Plot spikes for a batch on the given canvas and axes.

    Args:
        spikes (torch.Tensor): Spikes data with shape [num_neurons, num_spike_index].
        num_neurons (int): Number of neurons (channels).
        num_spike_index (int): Number of spike indices (time steps).
        canvas (FigureCanvasTkAgg): The canvas to render the plot.
        ax (matplotlib.axes.Axes): The axes to render the plot.
    """
    # Clear previous plots
    ax.clear()
    print(f"shape of spikes {spikes.shape}")
    spikes = spikes[0, 0]  # First batch, removing the channel dimension
    # Retrieve spike coordinates
    y_coords, x_coords = torch.nonzero(spikes, as_tuple=True)  # Non-zero indices for spikes
    ax.scatter(x_coords, y_coords, s=1, c="black", label="Spikes")  # Scatter plot for spikes

    # Calculate average spike rate per neuron
    avg_spike_rates = spikes.sum(dim=1).float() / num_spike_index
    max_spike_rate = avg_spike_rates.max()
    max_spike_channel = avg_spike_rates.argmax()
    
    print(f"avrg spike rate:{avg_spike_rates}, max spike rate:{max_spike_rate}, max spiking channel: {max_spike_channel}")

    # Annotate the plot with the highest average spike information
    ax.annotate(
        f"Neuron {max_spike_channel + 1} has max avg spikes: {max_spike_rate:.3f}",
        xy=(0.5, -0.1),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
    )

    # Customize y-ticks and labels
    yticks = np.arange(num_neurons)
    ytick_labels = [f"Neuron {i + 1}" for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylim(-0.5, num_neurons - 0.5)

    # Set x-axis limits
    ax.set_xlim(0, num_spike_index)
    ax.set_xlabel("Spike Event Index", fontsize=10)
    ax.set_ylabel("Neurons", fontsize=10)
    ax.set_title("Positive Spikes Visualization", fontsize=12)

    # Draw the canvas
    canvas.draw()
