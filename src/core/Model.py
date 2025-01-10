import torch
import torch.nn as nn
from snntorch import utils
from snntorch import surrogate,utils
import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np


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
    def get_parameters(self):
        return getattr(self, "net", self).parameters()




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
        # beta_idep = torch.rand((self.num_hidden), dtype = torch.float)
        # print(self.num_hidden)
        beta_idep_out = torch.rand((self.num_outputs), dtype = torch.float)
        # beta_indep = torch.rand(1)
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
    def get_parameters(self):
        return getattr(self, "net", self).parameters()


class SNNModel_population:
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
        - diff from default: population variable hardcoded
        """
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.betaLIF = betaLIF
        self.tresholdLIF = tresholdLIF
        self.device = device

        population=10

        # Surrogate gradient for spiking neuron
        spike_grad = surrogate.fast_sigmoid()

        # Define the SNN
        self.net = nn.Sequential(
            nn.Linear(self.num_inputs, self.num_hidden),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF),
            nn.Linear(self.num_hidden, self.num_hidden),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF),
            nn.Linear(self.num_hidden, self.num_hidden),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF),            
            nn.Linear(self.num_hidden, self.num_outputs
                      *population),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF, output=True)
        ).to(self.device)
        
    def forward(self, data, timestep):
        spk_rec = []
        utils.reset(self.net)  # Resets hidden states for all LIF neurons in the network
        # print(f"data shape{data.shape}")
        for step in range(timestep):
            spk_out, _ = self.net(data[step])  # Forward pass for one time step
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)  # Shape: [timestep, batch_size, output_size]
    def get_parameters(self):
        return getattr(self, "net", self).parameters()


class SNNModel_droput:
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
            nn.Dropout(p=0.3),  
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.Dropout(p=0.3),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF),
            nn.Linear(self.num_hidden, self.num_outputs),
            nn.Dropout(p=0.3),
            snn.Leaky(beta=self.betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=self.tresholdLIF, output=True)
        ).to(self.device)

##################### to be del: at batch accuracy
# spk_rec = []
#   snn_utils.reset(net)  # resets hidden states for all LIF neurons in net
#   for step in range(timestep):
#       spk_out, mem_out = net(data[step])
#       spk_rec.append(spk_out)
#  return torch.stack(spk_rec)
#####################


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
    
    def get_parameters(self):
        return getattr(self, "net", self).parameters()

#####Maintaing compatibility with old code DO THIS:
# Update to BaseSNNModel

# class BaseSNNModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def get_net(self):
#         return getattr(self, "net", self)

# Training Code Compatibility

# Replace model.net.eval() with:

# model.get_net().eval()

# Replace model.net.parameters() with:

# model.get_net().parameters()

# Replace model.net.forward(data) with:

# model.forward(data)

# This ensures seamless compatibility with both model implementations.

##############################################

class BaseSNNModel(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, tresholdLIF, device):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.tresholdLIF = tresholdLIF
        self.device = device

        #the important self.net
        # self.net = None  # this leaves None in custom forward

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method must be implemented in the derived class.")

    # def get_parameters(self):
    #     return super().parameters()  # Correctly registers all layers    
    def get_parameters(self):
        return self.parameters()  # Works for both models

    def get_state_dict(self):
        return self.net.state_dict()
    # usage
    # model.get_net().state_dict()  # Works for both old and new models

    # def get_state_dict(self):
    #     if hasattr(self, "net"):
    #         return self.net.state_dict()  # Old models
    #     else:
    #         return super().state_dict()  # New models     
    # usgae example:
    # "model_state_dict": self.model.get_state_dict() if self.model else None,
    
    def get_net(self):
        return getattr(self, "net", self)   
    
    # def get_parameters(self):
    #     if self.net is not None:
    #         return self.net.parameters()  # For models with self.net defined
    #     return super().parameters()       # For models managing layers manually
    # # usage:
    # # model.get_parameters() 
    
    def train(self, mode=True):
        super().train(mode)  # This sets the training mode for all registered layers
        self.training = mode  # Ensure compatibility if needed    
    
    # def train(self, mode=True):
    #     for layer in self.net:
    #         if isinstance(layer, nn.Module):
    #             layer.train(mode)
    #     self.training = mode

    def load_state_dict(self, state_dict, strict=True):
        if hasattr(self, "net"):
            self.net.load_state_dict(state_dict, strict)
        else:
            super().load_state_dict(state_dict, strict)
        
    # def load_state_dict(self, state_dict):
    #     if hasattr(self, "net"):
    #         # For old models using nn.Sequential
    #         self.net.load_state_dict(state_dict)
    #     else:
    #         # For new models with custom layers
    #         super().load_state_dict(state_dict) 
    def reset_states(self):
        """Reset states for layers with state management."""
        if hasattr(self, "layers"):
            for layer in self.layers:
                if hasattr(layer, "reset_state"):
                    layer.reset_state()

#learnable
class RDL_SNNModel(BaseSNNModel):
    def __init__(self, num_inputs, num_hidden, num_outputs, betaLIF, tresholdLIF, device, num_layers=4, random_learning=False):
        super().__init__(num_inputs, num_hidden, num_outputs, tresholdLIF, device)

        if random_learning:
            beta1 = torch.rand(num_hidden)
            threshold1 = torch.rand(num_hidden)
            beta2 = torch.rand(num_outputs)
            threshold2 = torch.rand(num_outputs)
        else:
            beta1 = torch.full((num_hidden,), betaLIF)
            beta2 = torch.full((num_outputs,), betaLIF)
            threshold1 = torch.full((num_hidden,), tresholdLIF)
            threshold2 = torch.full((num_outputs,), tresholdLIF)        
        

        spike_grad = surrogate.fast_sigmoid()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(num_inputs, num_hidden).to(device))
        self.layers.append(snn.Leaky(beta=beta1, learn_beta=True, learn_threshold=True, spike_grad=spike_grad, init_hidden=False, threshold=threshold1).to(device))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(num_hidden, num_hidden).to(device))
            self.layers.append(snn.Leaky(beta=beta1, learn_beta=True, learn_threshold=True, spike_grad=spike_grad, init_hidden=False, threshold=threshold1).to(device))

        self.layers.append(nn.Linear(num_hidden, num_outputs).to(device))
        self.layers.append(snn.Leaky(beta=beta2, learn_beta=True, learn_threshold=True, spike_grad=spike_grad, init_hidden=False, threshold=threshold2, output=True).to(device))
        
        # Move model to the correct device
        # self.to(self.device)

    def forward(self, data, timestep):
        data = data.to(self.device)  # Ensure data is on the correct device
        spk_rec = []
        # mem = [None] * (len(self.layers) // 2)
        # Initialize Membrane States at Each Forward Pass
        mem = [layer.init_leaky() for layer in self.layers if isinstance(layer, snn.Leaky)]

        for step in range(timestep):
            x = data[step]
            for idx in range(0, len(self.layers), 2):
                x = self.layers[idx](x)
                x, mem[idx//2] = self.layers[idx+1](x, mem[idx//2])

            # Detach after each timestep to prevent graph buildup
            spk_rec.append(x)

        return torch.stack(spk_rec)

    
class RD_SNNModel(BaseSNNModel):
    def __init__(self, num_inputs, num_hidden, num_outputs, betaLIF, tresholdLIF, device, num_layers=4):
        super().__init__(num_inputs, num_hidden, num_outputs, tresholdLIF, device)
        self.betaLIF = betaLIF

        spike_grad = surrogate.fast_sigmoid()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(num_inputs, num_hidden).to(device))
        self.layers.append(snn.Leaky(beta=betaLIF, spike_grad=spike_grad, init_hidden=False, threshold=tresholdLIF).to(device))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(num_hidden, num_hidden).to(device))
            self.layers.append(snn.Leaky(beta=betaLIF, spike_grad=spike_grad, init_hidden=False, threshold=tresholdLIF).to(device))

        self.layers.append(nn.Linear(num_hidden, num_outputs).to(device))
        self.layers.append(snn.Leaky(beta=betaLIF, spike_grad=spike_grad, init_hidden=False, threshold=tresholdLIF, output=True).to(device))
        
        # Move model to the correct device
        # self.to(self.device)

    def forward(self, data, timestep):
        data = data.to(self.device)  # Ensure data is on the correct device
        spk_rec = []
        # mem = [None] * (len(self.layers) // 2)
        # Initialize Membrane States at Each Forward Pass
        mem = [layer.init_leaky() for layer in self.layers if isinstance(layer, snn.Leaky)]

        for step in range(timestep):
            x = data[step]
            for idx in range(0, len(self.layers), 2):
                x = self.layers[idx](x)
                x, mem[idx//2] = self.layers[idx+1](x, mem[idx//2])

            # Detach after each timestep to prevent graph buildup
            spk_rec.append(x)

        return torch.stack(spk_rec)
    
import torch
import torch.nn as nn
import snntorch as snn

class RD_SNNModel_Synaptic(BaseSNNModel):
    def __init__(self, num_inputs, num_hidden, num_outputs, tresholdLIF, device, num_layers=4, alpha=0.9, betaLIF=0.8):
        super().__init__(num_inputs, num_hidden, num_outputs, tresholdLIF, device)

        spike_grad = snn.surrogate.fast_sigmoid()

        # Use ModuleList to register layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, num_hidden))
        self.layers.append(snn.Synaptic(alpha=alpha, beta=betaLIF, spike_grad=spike_grad, threshold=tresholdLIF))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
            self.layers.append(snn.Synaptic(alpha=alpha, beta=betaLIF, spike_grad=spike_grad, threshold=tresholdLIF))

        self.layers.append(nn.Linear(num_hidden, num_outputs))
        self.layers.append(snn.Synaptic(alpha=alpha, beta=betaLIF, spike_grad=spike_grad, threshold=tresholdLIF, output=True))
        self.to(self.device)
    def forward(self, data, timestep):
        spk_rec = []
        # Initialize synaptic and membrane states correctly
        syn, mem = zip(*[layer.reset_mem() if isinstance(layer, snn.Synaptic) else (None, None) for layer in self.layers])
        syn, mem = list(syn), list(mem)


        for step in range(timestep):
            x = data[step]
            for idx in range(0, len(self.layers), 2):
                x = self.layers[idx](x)  # Corresponds to cur1 = self.fc1(x)
                x, syn[idx // 2], mem[idx // 2] = self.layers[idx + 1](x, syn[idx // 2], mem[idx // 2])  # spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            spk_rec.append(x)  # Collects spike output from the final layer

        return torch.stack(spk_rec)


class RDL_SNNModel_Synaptic(BaseSNNModel):
    def __init__(self, num_inputs, num_hidden, num_outputs, tresholdLIF, device, num_layers=4, alpha=0.9, betaLIF=0.8, random_learning=False):
        super().__init__(num_inputs, num_hidden, num_outputs, tresholdLIF, device)

        if random_learning:
            beta1 = torch.rand(num_hidden)
            alpha1 = torch.rand(num_hidden)
            threshold1 = torch.rand(num_hidden)
            beta2 = torch.rand(num_outputs)
            alpha2 = torch.rand(num_outputs)
            threshold2 = torch.rand(num_outputs)
        else:
            beta1 = torch.full((num_hidden,), betaLIF)
            alpha1 = torch.full((num_hidden,), betaLIF)
            beta2 = torch.full((num_outputs,), betaLIF)
            alpha2 = torch.full((num_outputs,), betaLIF)
            threshold1 = torch.full((num_hidden,), tresholdLIF)
            threshold2 = torch.full((num_outputs,), tresholdLIF)

        spike_grad = snn.surrogate.fast_sigmoid()

        # Use ModuleList to register layers
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(num_inputs, num_hidden))
        self.layers.append(snn.Synaptic(alpha=alpha1, beta=beta1, learn_alpha=True, learn_beta=True, spike_grad=spike_grad, threshold=threshold1, learn_threshold=True))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
            self.layers.append(snn.Synaptic(alpha=alpha1, beta=beta1, learn_alpha=True, learn_beta=True, spike_grad=spike_grad, threshold=threshold1, learn_threshold=True))

        self.layers.append(nn.Linear(num_hidden, num_outputs))
        self.layers.append(snn.Synaptic(alpha=alpha2, beta=beta2, learn_alpha=True, learn_beta=True, spike_grad=spike_grad, threshold=threshold2, learn_threshold=True, output=True))
        self.to(self.device)
    def forward(self, data, timestep):
        spk_rec = []
        # Initialize synaptic and membrane states correctly
        syn, mem = zip(*[layer.reset_mem() if isinstance(layer, snn.Synaptic) else (None, None) for layer in self.layers])
        syn, mem = list(syn), list(mem)


        for step in range(timestep):
            x = data[step]
            for idx in range(0, len(self.layers), 2):
                x = self.layers[idx](x)  # Corresponds to cur1 = self.fc1(x)
                x, syn[idx // 2], mem[idx // 2] = self.layers[idx + 1](x, syn[idx // 2], mem[idx // 2])  # spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            spk_rec.append(x)  # Collects spike output from the final layer

        return torch.stack(spk_rec)


# old not real dynamic assumed
class DynamicSNNModel(BaseSNNModel):
    def __init__(self, num_inputs, num_hidden, num_outputs, betaLIF, tresholdLIF, device, num_layers=4):
        super().__init__(num_inputs, num_hidden, num_outputs, tresholdLIF, device)
        self.betaLIF = betaLIF

        # Surrogate gradient for spiking neuron
        spike_grad = surrogate.fast_sigmoid()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(num_inputs, num_hidden))
        layers.append(snn.Leaky(beta=betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=tresholdLIF))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(snn.Leaky(beta=betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=tresholdLIF))

        # Output layer
        layers.append(nn.Linear(num_hidden, num_outputs))
        layers.append(snn.Leaky(beta=betaLIF, spike_grad=spike_grad, init_hidden=True, threshold=tresholdLIF, output=True))

        self.net = nn.Sequential(*layers).to(device)

    def forward(self, data, timestep):
        spk_rec = []
        utils.reset(self.net)  # Resets hidden states for all LIF neurons in the network
        # print(f"data shape{data.shape}")
        for step in range(timestep):
            spk_out, _ = self.net(data[step])  # Forward pass for one time step
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)  # Shape: [timestep, batch_size, output_size]


# class RD_SNNModel_Synaptic(BaseSNNModel):
#     def __init__(self, num_inputs, num_hidden, num_outputs, tresholdLIF, device, num_layers=4, alpha=0.9, betaLIF=0.5):
#         super().__init__(num_inputs, num_hidden, num_outputs, tresholdLIF, device)
#         self.device = device
#         spike_grad = snn.surrogate.fast_sigmoid()

#         layers = []
#         layers.append(nn.Linear(num_inputs, num_hidden))
#         layers.append(snn.Synaptic(alpha=alpha, beta=betaLIF, spike_grad=spike_grad, threshold=tresholdLIF))

#         for _ in range(num_layers - 1):
#             layers.append(nn.Linear(num_hidden, num_hidden))
#             layers.append(snn.Synaptic(alpha=alpha, beta=betaLIF, spike_grad=spike_grad, threshold=tresholdLIF))

#         layers.append(nn.Linear(num_hidden, num_outputs))
#         layers.append(snn.Synaptic(alpha=alpha, beta=betaLIF, spike_grad=spike_grad, threshold=tresholdLIF, output=True))

#         # self.net = nn.Sequential(*layers).to(device)

#     def forward(self, data, timestep):
#         spk_rec = []
#         syn = [None] * (len(self.net) // 2)
#         mem = [None] * (len(self.net) // 2)

#         for step in range(timestep):
#             x = data[step]
#             for idx in range(0, len(self.net), 2):
#                 x = self.net[idx](x)
#                 x, syn[idx // 2], mem[idx // 2] = self.net[idx + 1](x, syn[idx // 2], mem[idx // 2])
#             spk_rec.append(x)

#         return torch.stack(spk_rec)


# class DynamicSNNModel_Synaptic():
#     def __init__(self, num_inputs, num_hidden, num_outputs, tresholdLIF, device, num_layers=4, alpha=0.9, betaLIF=0.5):
#         self.num_inputs = num_inputs
#         self.num_hidden = num_hidden
#         self.num_outputs = num_outputs
#         self.alpha = alpha
#         self.betaLIF = betaLIF
#         self.tresholdLIF = tresholdLIF
#         self.device = device

#         # Surrogate gradient for spiking neuron
#         spike_grad = surrogate.fast_sigmoid()
        
#         layers = []
        
#         # Input layer
#         layers.append(nn.Linear(num_inputs, num_hidden))
#         layers.append(snn.Synaptic(alpha=self.alpha, beta=self.betaLIF, spike_grad=spike_grad, threshold=self.tresholdLIF))

#         # Hidden layers
#         for _ in range(num_layers - 1):
#             layers.append(nn.Linear(num_hidden, num_hidden))
#             layers.append(snn.Synaptic(alpha=self.alpha, beta=self.betaLIF, spike_grad=spike_grad, threshold=self.tresholdLIF))

#         # Output layer
#         layers.append(nn.Linear(num_hidden, num_outputs))
#         layers.append(snn.Synaptic(alpha=self.alpha, beta=self.betaLIF, spike_grad=spike_grad, threshold=self.tresholdLIF, output=True))

#         self.net = nn.Sequential(*layers).to(device)

#     def forward(self, data, timestep):
#         spk_rec = []
#         utils.reset(self.net)  # Resets all internal states for Synaptic neurons
        
#         first_layer = list(self.net.children())[0]
        
#         print("print the net",self.net)
        
#         for step in range(timestep):
#             print(f"data[step] type: {type(data[step])}, shape: {data[step].shape}")
#             print(f"First Layer Type: {type(first_layer)}, Output: {first_layer(data[step])}")
#             spk_out, *_ = self.net(data[step])  # Forward pass for one time step
#             spk_rec.append(spk_out)

#         return torch.stack(spk_rec)  # Shape: [timestep, batch_size, output_size]


# class DynamicSNNModel_Synaptic():
#     def __init__(self, num_inputs, num_hidden, num_outputs, betaLIF, tresholdLIF, device, num_layers=4, alpha=0.9):
#         self.num_inputs = num_inputs
#         self.num_hidden = num_hidden
#         self.num_outputs = num_outputs
#         self.alpha = alpha  # Synaptic current decay rate
#         self.betaLIF = betaLIF
#         self.tresholdLIF = tresholdLIF
#         self.device = device

#         # Surrogate gradient for spiking neuron
#         spike_grad = surrogate.fast_sigmoid()

#         layers = []

#         # Input layer with Synaptic neurons (no alpha or beta)
#         layers.append(nn.Linear(num_inputs, num_hidden))
#         layers.append(snn.Synaptic(alpha=self.alpha, beta=self.betaLIF,  spike_grad=spike_grad, threshold=tresholdLIF))

#         # Hidden layers with Synaptic neurons
#         for _ in range(num_layers - 1):
#             layers.append(nn.Linear(num_hidden, num_hidden))
#             layers.append(snn.Synaptic(alpha=self.alpha, beta=self.betaLIF, spike_grad=spike_grad, threshold=tresholdLIF))

#         # Output layer with Synaptic neurons
#         layers.append(nn.Linear(num_hidden, num_outputs))
#         layers.append(snn.Synaptic(alpha=self.alpha, beta=self.betaLIF, spike_grad=spike_grad, threshold=tresholdLIF, output=True))

#         self.net = nn.Sequential(*layers).to(device)

#     def forward(self, data, timestep):
#         spk_rec = []
#         utils.reset(self.net)
        
#         for step in range(timestep):
#             # Pass data through the network, which includes synaptic current and membrane potential
#             spk_out, syn, mem = self.net(data[step], syn, mem)  # Forward pass for one timestep
#             spk_rec.append(spk_out)

#         return torch.stack(spk_rec)  # Shape: [timestep, batch_size, output_size]


# class DynamicSNNModel_Synaptic():
#     def __init__(self, num_inputs, num_hidden, num_outputs, betaLIF, tresholdLIF, device, num_layers=4, alpha=0.9, beta=0.5):
#         self.num_inputs = num_inputs
#         self.num_hidden = num_hidden
#         self.num_outputs = num_outputs
#         self.betaLIF = betaLIF
#         self.tresholdLIF = tresholdLIF
#         self.device = device

#         # Surrogate gradient for spiking neuron
#         spike_grad = surrogate.fast_sigmoid()
        
#         layers = []
        
#         # Input layer
#         layers.append(nn.Linear(num_inputs, num_hidden))
        
#         layers.append(snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=tresholdLIF))

#         # Hidden layers
#         for _ in range(num_layers - 1):
#             layers.append(nn.Linear(num_hidden, num_hidden))
#             layers.append(snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=tresholdLIF))

#         # Output layer
#         layers.append(nn.Linear(num_hidden, num_outputs))
#         layers.append(snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=tresholdLIF, output=True))

#         self.net = nn.Sequential(*layers).to(device)

#     def forward(self, data, timestep):
#         spk_rec = []
#         utils.reset(self.net)  # Resets hidden states for all LIF neurons in the network
#         # print(f"data shape{data.shape}")
#         for step in range(timestep):
#             spk_out, _ = self.net(data[step])  # Forward pass for one time step
#             spk_rec.append(spk_out)

#         return torch.stack(spk_rec)  # Shape: [timestep, batch_size, output_size]    

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