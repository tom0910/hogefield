import torch
import torch.nn as nn
from snntorch import utils
from snntorch import surrogate
import snntorch as snn


class SNNModel_dl:
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
        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.num_inputs, self.num_hidden),
                snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF)
            ),
            nn.Sequential(
                nn.Linear(self.num_hidden, self.num_hidden),
                snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF)
            ),
            nn.Sequential(
                nn.Linear(self.num_hidden, self.num_outputs),
                snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF, output=True)
            )
        ]).to(self.device)

    def apply_input_delay(self, data, delay_steps):
        """
        Applies delays to the input data along the time axis.

        Parameters:
        - data: Input tensor of shape [timestep, batch_size, input_size].
        - delay_steps: List of delay steps for each input neuron.

        Returns:
        - Delayed input tensor of the same shape as data.
        """
        delayed_data = torch.zeros_like(data).to(self.device)
        for neuron_idx, delay in enumerate(delay_steps):
            if delay < data.size(0):  # Ensure delay is within valid range
                delayed_data[delay:, :, neuron_idx] = data[:data.size(0) - delay, :, neuron_idx]
        return delayed_data

    def forward(self, data, timestep, delay_buffer=None):
        """
        Performs the forward pass of the SNN over a given number of time steps, incorporating delays.

        Parameters:
        - data: Input tensor of shape [timestep, batch_size, input_size].
        - timestep: Number of time steps to run the forward pass.
        - delay_buffer: Buffer to store delayed spikes (initialized as None).

        Returns:
        - Tensor of shape [timestep, batch_size, output_size].
        """
        spk_rec = []
        if delay_buffer is None:
            delay_buffer = [
                torch.zeros(timestep, data.size(1), self.num_hidden).to(self.device)
                for _ in range(len(self.net))
            ]

        # Reset hidden states for all LIF neurons
        for module in self.net:
            utils.reset(module)

        for step in range(timestep):
            current_input = data[step]

            # Process through layers with delays
            for layer_idx, layer in enumerate(self.net):
                delayed_spike = delay_buffer[layer_idx][step % timestep]
                current_input = current_input + delayed_spike  # Add delayed spikes
                spk_out, _ = layer(current_input)
                delay_buffer[layer_idx][(step + 1) % timestep] = spk_out  # Update buffer
                current_input = spk_out

            spk_rec.append(spk_out)

        return torch.stack(spk_rec)  # Shape: [timestep, batch_size, output_size]



# Define Model Parameters (GLOBAL values should be passed here)
class SNNModel:
    def __init__(self, num_inputs, num_hidden, num_outputs, betaLIF, tresholdLIF, device):
                #  , learning_rate):
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
        # self.learning_rate = learning_rate

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
        # print(f"data shape{data.shape}")
        for step in range(timestep):
            # print("Input Data's' shape:", data.shape) #Input Data's' shape: torch.Size([201, 64, 80]) = [time_step, batch_size, neural_number], neural_number is iput size
            # print("data.size(0) is:",data.size(0)) # data.size(0) is: 201 = time_steps
            # print("Step:", step)
            # print("Input Data[step]'s' Shape:", data[step].shape) #Input Data Shape: torch.Size([batch_size, neural_numbers]) = Input Data Shape: torch.Size([64, 80])
            # print_structure("net(data[step]):",net(data[step]))
            # print("net(data[step]):",self.net(data[step]).shape)
            spk_out, _ = self.net(data[step])  # Forward pass for one time step
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)  # Shape: [timestep, batch_size, output_size]
    
    # Optional: Visualization Utility (Specific to this model)

class SNNModel_population:
    def __init__(self, num_inputs, num_hidden, num_outputs, betaLIF, tresholdLIF, device):
                #  , learning_rate):
        """
        Initializes the Spiking Neural Network (SNN) with the given parameters.

        Parameters:
        - num_inputs: Number of input neurons.
        - num_hidden: Number of neurons in the hidden layers.
        - num_outputs: Number of output neurons.
        - betaLIF: Decay factor for Leaky Integrate-and-Fire neurons.
        - tresholdLIF: Firing threshold for LIF neurons.
        - device: Device to run the model (e.g., 'cpu' or 'cuda').
        - diff from default: population variable hardcoded
        """
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.betaLIF = betaLIF
        self.tresholdLIF = tresholdLIF
        self.device = device
        # self.learning_rate = learning_rate

        population=10

        # Surrogate gradient for spiking neuron
        spike_grad = surrogate.fast_sigmoid()

        # Define the SNN
        self.net = nn.Sequential(
            nn.Linear(self.num_inputs, self.num_hidden),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF),
            nn.Linear(self.num_hidden, self.num_hidden),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF),
            nn.Linear(self.num_hidden, self.num_outputs
                      *population),
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
        # print(f"data shape{data.shape}")
        for step in range(timestep):
            # print("Input Data's' shape:", data.shape) #Input Data's' shape: torch.Size([201, 64, 80]) = [time_step, batch_size, neural_number], neural_number is iput size
            # print("data.size(0) is:",data.size(0)) # data.size(0) is: 201 = time_steps
            # print("Step:", step)
            # print("Input Data[step]'s' Shape:", data[step].shape) #Input Data Shape: torch.Size([batch_size, neural_numbers]) = Input Data Shape: torch.Size([64, 80])
            # print_structure("net(data[step]):",net(data[step]))
            # print("net(data[step]):",self.net(data[step]).shape)
            spk_out, _ = self.net(data[step])  # Forward pass for one time step
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)  # Shape: [timestep, batch_size, output_size]

class SNNModel_droput:
    def __init__(self, num_inputs, num_hidden, num_outputs, betaLIF, tresholdLIF, device):
                #  , learning_rate):
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
        # self.learning_rate = learning_rate

        # Surrogate gradient for spiking neuron
        spike_grad = surrogate.fast_sigmoid()

        # Define the SNN
        self.net = nn.Sequential(
            nn.Linear(self.num_inputs, self.num_hidden),
            nn.Dropout(p=0.3),  
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.Dropout(p=0.3),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF),
            nn.Linear(self.num_hidden, self.num_outputs),
            nn.Dropout(p=0.3),
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
        # print(f"data shape{data.shape}")
        for step in range(timestep):
            # print("Input Data's' shape:", data.shape) #Input Data's' shape: torch.Size([201, 64, 80]) = [time_step, batch_size, neural_number], neural_number is iput size
            # print("data.size(0) is:",data.size(0)) # data.size(0) is: 201 = time_steps
            # print("Step:", step)
            # print("Input Data[step]'s' Shape:", data[step].shape) #Input Data Shape: torch.Size([batch_size, neural_numbers]) = Input Data Shape: torch.Size([64, 80])
            # print_structure("net(data[step]):",net(data[step]))
            # print("net(data[step]):",self.net(data[step]).shape)
            spk_out, _ = self.net(data[step])  # Forward pass for one time step
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)  # Shape: [timestep, batch_size, output_size]
    
    # Optional: Visualization Utility (Specific to this model)
    
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