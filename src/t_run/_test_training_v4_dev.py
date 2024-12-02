import os
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from utils.training_functional import prepare_dataset
from core.Model import SNNModel_population
from core.CheckPoinManager import CheckpointManager
from utils.training_utils import train_population
from snntorch import functional as SNNF

def select_pth_and_dir(initial_dir="/project/hyperparam"):
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

def main_train():
    # Select file and directory
    # file_path, dir_path = select_pth_and_dir()
    # if not file_path or not dir_path:
    #     print("File or directory not selected. Exiting.")
    #     return

    # Setup directories
    # log_file = "output.log"
    # log_path = os.path.join(dir_path, log_file)
    # plots_dir = os.path.join(dir_path, "plots")
    # pth_dir = os.path.join(dir_path, "pth")
    
    file_path="/project/hyperparam/pop_trial/snn_model_hogefield.pth"
    dir_path="/project/hyperparam/pop_trial"

    # Load checkpoint
    chp_manager = CheckpointManager.load_checkpoint_with_defaults(file_path=file_path)
    params = chp_manager.get_hyperparameters()

    # Preprocess dataset by batch size
    train_loader, test_loader = prepare_dataset(pth_file_path=file_path, params=params)

    # Define loss function
    loss_fn = SNNF.mse_count_loss(correct_rate=0.75, incorrect_rate=0.25, population_code=True, num_classes=35)
    
    # Align keys with SNNModel requirements
    paramsSNN = {
        "num_inputs":       params.get("number_of_inputs", 16),
        "num_hidden":       params.get("number_of_hidden_neurons", 256),
        "num_outputs":      params.get("number_of_outputs", 35),
        "beta_lif":         params.get("beta_lif", 0.9),  # Default if missing
        "threshold_lif":    params.get("threshold_lif", 0.5),  # Default if missing
        "device":           params.get("device", "cuda"),
    }


    # Initialize model
    model = SNNModel_population(
        num_inputs      = paramsSNN["num_inputs"],
        num_hidden      = paramsSNN["num_hidden"],
        num_outputs     = paramsSNN["num_outputs"],
        betaLIF         = paramsSNN["beta_lif"],
        tresholdLIF     = paramsSNN["threshold_lif"],  # Fixed typo from "treshold"
        device          = paramsSNN["device"],
    )

    chp_manager.print_contents()
    # overwrite the Checkpointmanager with model has before 
    chp_manager = CheckpointManager.load_checkpoint_with_model(file_path=file_path, model=model)
    chp_manager.print_contents()
    
    # Train the model
    train_population  (
        checkpoint_mngr=chp_manager,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        num_epochs=100,
        checkpoint_dir=pth_dir,
        plots_dir=plots_dir,
    )

if __name__ == "__main__":
    main_train()
