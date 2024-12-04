import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import config.config as config
import utils.training_functional as TF
import subprocess
import pickle
import json
from datetime import datetime
from types import SimpleNamespace
import re
import torch
import shutil
from core.Model import SNNModel  
from core.CheckPoinManager import CheckpointManager  # Adjust the import path as needed
from utils.training_functional import prepare_dataset
from utils.training_utils import train_as_hypp_change
from snntorch import functional as SNNF

import os
def main_train_v1_1(pth_saved_dir,DEFAULT_DIR):
    
    
    # Ensure pth_save_path is valid before checking with os.path.exists
    dir = pth_saved_dir if pth_saved_dir and os.path.exists(pth_saved_dir) else DEFAULT_DIR
    # Select file and directory
    file_path, dir_path = select_pth_and_dir(initial_dir=dir)

    if not file_path or not dir_path:
        print("File or directory not selected. Exiting.")
        return

    # Setup directories
    log_file = "output.log"
    log_path = os.path.join(dir_path, log_file)
    plots_dir = os.path.join(dir_path, "plots")
    pth_dir = os.path.join(dir_path, "pth")

    # Load checkpoint
    chp_manager = CheckpointManager.load_checkpoint_with_defaults_v1_1(file_path=file_path)  # Use the v1.1 function
    params = chp_manager.get_hyperparameters()

    # Align keys for the model class
    try:
        # Dynamically determine model type
        model_type = params.get("model_type", "SNNModel")  # Default to SNNModel if not specified
        
        model_hyperparams = {
            "num_inputs": params.get("number_of_inputs"),
            "num_hidden": params.get("number_of_hidden_neurons"),
            "num_outputs": params.get("number_of_outputs"),
            "betaLIF": params.get("beta_lif"),
            "tresholdLIF": params.get("threshold_lif"),
            "device": params.get("device"),
        }
    except KeyError as e:
        raise ValueError(f"Missing required parameter from checkpoint (pth): {e}")

    # Dynamically initialize the model
    model = CheckpointManager.MODEL_REGISTRY[model_type](**model_hyperparams)

    # Overwrite the CheckpointManager with the correct model instance
    chp_manager = CheckpointManager.load_checkpoint_with_model(file_path=file_path, model=model)

    # Preprocess dataset by batch size
    train_loader, test_loader = prepare_dataset(pth_file_path=file_path, params=params)

    population_code = model_type == "SNNModel_population"
    num_classes = params.get("number_of_outputs", 35)  # Original number of outputs

    correct_rate= params.get("correct rate",1)
    incorrect_rate= params.get("correct rate",0)
    loss_fn = SNNF.mse_count_loss(
        correct_rate=correct_rate,
        incorrect_rate=incorrect_rate,
        population_code=population_code,
        num_classes=num_classes  # remains compatible with SNNModel
    )

    # Train the model
    train_as_hypp_change(
        checkpoint_mngr=chp_manager,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        num_epochs=100,
        checkpoint_dir=pth_dir,
        plots_dir=plots_dir,
    )
    
def select_pth_and_dir(initial_dir):
    """
    Select a .pth file and a directory using Tkinter dialogs.
    Returns the file path and directory path.
    """
    # Ensure the initial directory exists
    if not os.path.exists(initial_dir):
        os.makedirs(initial_dir)

    # Tkinter root setup
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Select the .pth file
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Select a .pth File",
        filetypes=[("PyTorch Files", "*.pth"), ("All Files", "*.*")]
    )
    if not file_path:
        messagebox.showwarning("No File Selected", "Please select a valid .pth file.")
        return None, None

    # Select or create a directory
    selected_dir = filedialog.askdirectory(initialdir=initial_dir, title="Select or Create Directory")
    if not selected_dir:
        messagebox.showinfo("Cancelled", "No directory selected.")
        return file_path, None

    if not os.path.exists(selected_dir):
        create = messagebox.askyesno("Directory does not exist",
                                    f"The directory '{selected_dir}' does not exist. Do you want to create it?")
        if create:
            try:
                os.makedirs(selected_dir)
                messagebox.showinfo("Success", f"Directory '{selected_dir}' created successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create directory: {e}")
                return file_path, None
        else:
            return file_path, None

    return file_path, selected_dir

# Initialize global variables
pth_saved_dir = None  # Set to None initially
DEFAULT_DIR = "/project/hypertrain"    
if __name__ == "__main__":
    # Ensure pth_saved_dir is set
    if not pth_saved_dir:
        root = tk.Tk()
        root.withdraw()  # Hide main Tkinter window
        pth_saved_dir = filedialog.askdirectory(title="Select Saved Directory")
        if not pth_saved_dir:
            messagebox.showerror("Error", "No directory selected. Exiting.")
            exit()

    main_train_v1_1(pth_saved_dir, DEFAULT_DIR)