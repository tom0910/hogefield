import torch
import torchaudio.transforms as T
from torchaudio.transforms import AmplitudeToDB
from libs.SNNnanosenpy import SubsetSC, get_unique_labels, pad_sequence, label_to_index
import utils.functional as FU


def prepare_datasets(data_path="/project/data/GSC/", target_labels=None, get_labels=False, check_labels=False):
    """
    Prepare the training and testing datasets.

    Parameters:
    - data_path (str): Path to the dataset directory.
    - target_labels (list): Predefined target labels. If None, uses a default set.
    - get_labels (bool): If True, retrieves unique labels from the training set.
    - check_labels (bool): If True, prints unique labels.

    Returns:
    - train_set (SubsetSC): Training dataset.
    - test_set (SubsetSC): Testing dataset.
    - target_labels (list): List of target labels.
    """
    # Default target labels if none are provided
    if target_labels is None:
        target_labels = [
            'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
            'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
            'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
            'visual', 'wow', 'yes', 'zero'
        ]
    
    # Create training and testing datasets
    train_set = SubsetSC(directory=data_path, subset="training", download=False)
    test_set = SubsetSC(directory=data_path, subset="testing", download=False)

    # Retrieve unique labels if requested
    if get_labels:
        target_labels = get_unique_labels(train_set)
    
    # Optionally print the labels
    if check_labels:
        print("Target labels:", target_labels)

    return train_set, test_set, target_labels

def parse_hyperparameter_file(file_path):
    """
    Parse hyperparameter file and return a dictionary of parameters.

    Parameters:
    - file_path (str): Path to the hyperparameters file.

    Returns:
    - params (dict): Dictionary of hyperparameters.
    """
    params = {}
    with open(file_path, "r") as file:
        for line in file:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")  # Normalize keys
                value = value.strip()
                if value.replace(".", "", 1).isdigit():  # Convert to int/float
                    value = float(value) if "." in value else int(value)
                params[key] = value
    return params


import torch
import torchaudio.transforms as T
import libs.SNNnanosenpy as snnnspy
from torchaudio.transforms import Spectrogram
from core.CustomMelScale import CustomMelScale

import torch
import re

def load_parameters_from_pth(file_path):
    """
    Load and standardize hyperparameters from a .pth file.

    Parameters:
    - file_path (str): Path to the .pth file.

    Returns:
    - params (dict): Extracted and standardized hyperparameters.
    """
    if not file_path.endswith(".pth"):
        raise ValueError("Only .pth files are supported.")

    try:
        # Load checkpoint
        checkpoint = torch.load(file_path)

        # Check for hyperparameters
        if "hyperparameters" not in checkpoint:
            raise ValueError("The .pth file does not contain 'hyperparameters' key.")

        params = checkpoint["hyperparameters"]

        # Standardize key names (e.g., "Beta (LIF)" -> "beta_lif")
        standardized_params = {
            re.sub(r"[()\s]+", "_", key).strip("_").lower(): value
            for key, value in params.items()
        }

        return standardized_params

    except FileNotFoundError:
        raise ValueError(f"The file {file_path} was not found.")
    except Exception as e:
        raise ValueError(f"Failed to parse .pth file: {e}")


# def load_parameters_from_pth(file_path):
#     """
#     Load parameters from a .pth file.

#     Parameters:
#     - file_path (str): Path to the .pth file.

#     Returns:
#     - params (dict): Extracted hyperparameters.
#     """
#     # print(f"file_path: {file_path} (type: {type(file_path)})")
#     if not file_path.endswith(".pth"):
#         raise ValueError("Only .pth files are supported.")

#     try:
#         checkpoint = torch.load(file_path)
#         if "hyperparameters" in checkpoint:
#             params = checkpoint["hyperparameters"]
#             # print("Extracted Parameters from .pth file:")
#             # for key, value in params.items():
#             #     print(f"{key}: {value}")
#             return params
#         else:
#             raise ValueError("The .pth file does not contain hyperparameters.")
#     except Exception as e:
#         raise ValueError(f"Failed to parse .pth file: {e}")

def pad_sequence(batch):
    # Ensure all items are tensors and have at least 2 dimensions
    batch = [item.t() if item.ndim == 2 else item.unsqueeze(0) for item in batch]
    # Pad sequences
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    # Permute back
    return batch.permute(0, 2, 1)

import utils.preprocess_collate as collate

def custom_collate_fn(batch, params, target_labels, pth_file_path):
    """
    Custom collate function for DataLoader to process audio batches.
    """
    # Ensure parameters are loaded
    params = load_parameters_from_pth(pth_file_path)
    try:
        n_fft = int(params["n_fft"])
        hop_length = int(params["hop_length"])
        n_mels = int(params["n_mels"])
        f_min = float(params["f_min"])
        f_max = float(params["f_max"])
        threshold = float(params["sf_threshold"])  # Explicit conversion
        filter = params["filter_type_custom_or_standard"]
        sample_rate = int(params["wav_file_samples"])        
        # n_fft = params["n_fft"]
        # hop_length = params["hop_length"]
        # n_mels = params["n_mels"]
        # f_min = params["f_min"]
        # f_max = params["f_max"]
        # threshold = params["sf_threshold"]
        # filter = params["filter_type_custom_or_standard"]
        # sample_rate = params["wav_file_samples"]
    except KeyError as e:
        raise ValueError(f"Missing required parameter: {e}")

    # print(f" read and used from pth => n_fft: {n_fft}, hop_length: {hop_length}, n_mels: {n_mels}, f_min: {f_min}, f_max: {f_max}, threshold: {threshold}")

    tensors, targets = [], []

    # Collect waveforms and labels
    for waveform, _, label, *_ in batch:
        tensors.append(waveform)
        targets.append(label_to_index(label, target_labels))

    spikes, new_targets, num_neurons, base_cums = collate.preprocess_collate(
        tensors=tensors, targets=targets, n_fft=n_fft, hop_length=hop_length, 
        n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max, threshold=threshold, filter=filter
        )
    return spikes, new_targets, num_neurons, base_cums 
    
    # # Pad the sequence of tensors
    # tensors = pad_sequence(tensors)
    
    # # Spectrogram transformation
    # spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0,center=True) # center: Adds (n_fft // 2) zero-padding on both sides of the input signal, which can ensure alignment and better frequency analysis at the edges.
    # spectrogram = spectrogram_transform(tensors)

    # # Initialize CustomMelScale with the filter type and configuration
    # custom_mel_scale = CustomMelScale(
    #     n_mels=n_mels,
    #     sample_rate=sample_rate,
    #     f_min=f_min,
    #     f_max=f_max,
    #     n_stft=n_fft // 2 + 1,
    #     filter_type=filter
    # )

    # # Apply the CustomMelScale transformation
    # mel_spectrogram = custom_mel_scale(spectrogram)
    # original_min, original_max = mel_spectrogram.min(), mel_spectrogram.max()
    # mel_spectrogram_normalized = FU.normalize(mel_spectrogram, normalize_to=1)

    # # Time-step adjustment
    # delta_t = 1 / sample_rate
    # csum = torch.cumsum(mel_spectrogram_normalized, dim=-1) * delta_t

    # # Step forward encoding
    # base_cums, pos_accum, neg_accum = FU.step_forward_encoding(csum, threshold, neg=True)
    # spikes = pos_accum
    
    # # print(f"spike shape:{spikes.shape}")

    # # Neuron and spike information
    # num_neurons = spikes.shape[2]  # Dimension for neurons
    # num_spike_index = spikes.shape[3]

    # # Stack targets into a tensor
    # targets = torch.tensor(targets)

    # return spikes, targets, num_neurons, base_cums


# def load_pth_file(self):
#     """
#     Open file dialog to load a .pth file and extract hyperparameters.
#     """
#     file_path = filedialog.askopenfilename(
#         title="Select a .pth File",
#         filetypes=(("PTH Files", "*.pth"), ("All Files", "*.*"))
#     )
#     if not file_path:
#         self.status_label.config(text="Status: No file selected.")
#         return
    
#     try:
#         self.params = load_parameters_from_pth(file_path)
#         self.status_label.config(text=f"Status: Loaded parameters from {file_path}")
#         print("Loaded Parameters:")
#         for key, value in self.params.items():
#             print(f"{key}: {value}")
#         messagebox.showinfo("Success", "Parameters loaded successfully!")
#     except Exception as e:
#         self.status_label.config(text="Status: Failed to load parameters.")
#         messagebox.showerror("Error", f"Failed to load parameters: {e}")
            

import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from torch.utils.data import DataLoader
import libs.SNNnanosenpy as snnnspy
from utils.functional import step_forward_encoding, normalize
from torchaudio.transforms import Spectrogram
from core.CustomMelScale import CustomMelScale
from libs.SNNnanosenpy import SubsetSC, get_unique_labels, label_to_index
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

# from utils.training_functional import load_parameters_from_pth

class TestCollateApp:
    def __init__(self, parent):
        self.parent = parent
        self.parent.title("Collate Function Test")
        
        # Buttons
        self.load_pth_button = tk.Button(parent, text="Load .pth File", command=self.load_pth_file)
        self.load_pth_button.pack(padx=10, pady=5)

        self.test_collate_button = tk.Button(parent, text="Test Collate Function", command=self.test_collate_function)
        self.test_collate_button.pack(padx=10, pady=5)

        self.status_label = tk.Label(parent, text="Status: Waiting for input...")
        self.status_label.pack(padx=10, pady=10)
        
        # Matplotlib figure for spike visualization
        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(padx=10, pady=5)        
        
        # Initialize parameters and figure for plotting
        self.params = None
        self.pth_file_path = None
        self.last_spikes = None
        self.last_num_neurons = None
        self.last_num_spike_index = None

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)        
        
    def load_pth_file(self):
        """
        Open file dialog to load a .pth file and extract hyperparameters.
        """
        file_path = filedialog.askopenfilename(
            title="Select a .pth File",
            filetypes=(("PTH Files", "*.pth"), ("All Files", "*.*"))
        )
        if not file_path:  # Handle cancelation or invalid path
            self.status_label.config(text="Status: No file selected.")
            return

        try:
            # Ensure the file is a .pth file
            if not file_path.endswith(".pth"):
                raise ValueError("Invalid file type. Please select a .pth file.")

            self.params = load_parameters_from_pth(file_path)
            self.pth_file_path = file_path  # Save the file path for later use
            self.status_label.config(text=f"Status: Loaded parameters from {file_path}")
            print("Loaded Parameters:")
            for key, value in self.params.items():
                print(f"{key}: {value}")
            messagebox.showinfo("Success", "Parameters loaded successfully!")
        except Exception as e:
            self.status_label.config(text="Status: Failed to load parameters.")
            messagebox.showerror("Error", f"Failed to load parameters: {e}")
        
    def test_collate_function(self):
        """
        Test the custom collate function using real data and the loaded parameters.
        """
        try:
            # Ensure the parameters and .pth file are loaded
            if not self.params or not self.pth_file_path:
                raise ValueError("No .pth file or parameters loaded. Please load a .pth file first.")

            # Prepare datasets
            train_set, test_set, target_labels = prepare_datasets(
                data_path="/project/data/GSC/",
                check_labels=False
            )

            # Create DataLoader with custom_collate_fn
            train_loader = DataLoader(
                dataset=train_set,
                batch_size=int(self.params["batch_size"]),  # Use batch size from parameters
                collate_fn=lambda batch: custom_collate_fn(
                    batch=batch,
                    params=self.params,
                    target_labels=target_labels,
                    pth_file_path=self.pth_file_path
                )
            )

            # Iterate through a few batches to test functionality
            for batch_idx, (spikes, targets, neuron_number, base_cums) in enumerate(train_loader):
                print(f"\n--- Batch {batch_idx} ---")
                print(f"Spikes shape: {spikes.shape}")  # Neural encoding dimensions
                print(f"Targets shape: {targets.shape}")  # Shape of target labels
                print(f"Neuron number: {neuron_number}")  # Number of neurons
                print(f"Base cumulative sums shape: {base_cums.shape}")  # Cumulative sums
                
                # Plot spikes using matplotlib
                import display.batch_plotting as bplt
                bplt.plot_spikes(
                    spikes=spikes,
                    num_neurons=neuron_number,
                    num_spike_index=spikes.shape[3],
                    canvas=self.canvas,
                    ax=self.ax
                )
                # Break after a few batches to limit output
                if batch_idx == 2:
                    break

            # Success message for the UI
            self.status_label.config(text="Status: Collate function test succeeded!", fg="green")
            messagebox.showinfo("Success", "Collate function tested successfully!")
        except Exception as e:
            # Handle and display errors
            print(f"Collate function test failed: {e}")
            self.status_label.config(text="Status: Collate function test failed.", fg="red")
            messagebox.showerror("Error", f"Collate function test failed: {e}")


        
# Main Program For test only
if __name__ == "__main__":
    root = tk.Tk()
    app = TestCollateApp(root)
    root.mainloop()        
