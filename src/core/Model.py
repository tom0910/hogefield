import torch
import torch.nn as nn
from snntorch import utils
from snntorch import surrogate
import snntorch as snn

# Define Model Parameters (GLOBAL values should be passed here)
class SNNModel:
    def __init__(self, num_inputs, num_hidden, num_outputs, betaLIF, tresholdLIF, device):
        """
        Initializes the Spiking Neural Network (SNN) with the given parameters.

        Parameters:
        - num_inputs: Number of input neurons.
        - num_hidden: Number of neurons in the hidden layers.
        - num_outputs: Number of output neurons.
        - betaLIF: Decay factor for Leaky Integrate-and-Fire neurons.
        - tresholdLIF: Firing threshold for LIF neurons.
        - device: Device to run the model (e.g., 'cpu' or 'cuda').
        """
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.betaLIF = betaLIF
        self.tresholdLIF = tresholdLIF
        self.device = device

        # Surrogate gradient for spiking neuron
        spike_grad = surrogate.fast_sigmoid()

        # Define the SNN
        self.net = nn.Sequential(
            nn.Linear(self.num_inputs, self.num_hidden),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF),
            nn.Linear(self.num_hidden, self.num_hidden),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF),
            nn.Linear(self.num_hidden, self.num_outputs),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF, output=True)
        ).to(self.device)

    def forward(self, data, timestep):
        """
        Performs the forward pass of the SNN over a given number of time steps.

        Parameters:
        - data: Input tensor of shape [timestep, batch_size, input_size].
        - timestep: Number of time steps to run the forward pass.

        Returns:
        - Tensor of shape [timestep, batch_size, output_size].
        """
        spk_rec = []
        utils.reset(self.net)  # Resets hidden states for all LIF neurons in the network

        for step in range(timestep):
            spk_out, _ = self.net(data[step])  # Forward pass for one time step
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)  # Shape: [timestep, batch_size, output_size]
    
    # Optional: Visualization Utility (Specific to this model)


import matplotlib.pyplot as plt
import numpy as np

def plot_pipeline(tensor):
    """
    Plots a raster plot of the spiking activity for a given tensor.

    Parameters:
    - tensor: 3D Tensor of shape [timestep, batch_size, output_size].
    """
    print("Tensor shape:", tensor.shape)
    spike_data_sample = tensor[:, 0, :]  # Extract spiking activity for the first batch element
    print("Plotting shape:", spike_data_sample.shape)

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)

    # Move tensor to CPU if necessary
    spike_data_sample_cpu = spike_data_sample.cpu() if tensor.is_cuda else spike_data_sample

    # Convert tensor to NumPy array for plotting
    spike_data_sample_numpy = spike_data_sample_cpu.detach().numpy()

    # Create a raster plot
    ax.scatter(*np.where(spike_data_sample_numpy), s=1.5, c="black")

    plt.title("Input to SNN")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()