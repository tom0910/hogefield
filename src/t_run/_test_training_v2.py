from tkinter import filedialog, Tk, messagebox, simpledialog
import os
import tkinter as tk
from utils.training_functional import prepare_dataset, create_optimizer, load_latest_checkpoint, load_checkpoint
import utils.loading_functional as FL
from core.Model import SNNModel
from core.CheckPoinManager import CheckpointManager
from utils.training_utils import train, train_as_hypp_change
from snntorch import functional as SNNF 
import torch

initial_dir = "/project/hyperparam"

def choose_pth_file(initial_dir="/project/hyperparam"):
    """
    Open a dialog to select a .pth file and return its path.
    Handles cases where no file is selected.
    """
    # Ensure the initial directory exists
    if not os.path.exists(initial_dir):
        os.makedirs(initial_dir)

    # Open file dialog to select a .pth file
    selected_file = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Select a .pth File",
        filetypes=[("PyTorch Files", "*.pth"), ("All Files", "*.*")]
    )

    # Check if a file was selected
    if not selected_file:
        messagebox.showwarning("No File Selected", "Please select a valid .pth file.")
        return None  # Return None if no file was selected

    return selected_file


def select_or_create_directory(initial_dir="/project/hyperparam"):
    """
    Open a directory selection dialog allowing the user to select an existing directory
    or input a new directory name to create.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Open a directory dialog starting from the initial directory
    selected_dir = filedialog.askdirectory(initialdir=initial_dir, title="Select or Create Directory")
    
    if not selected_dir:
        messagebox.showinfo("Cancelled", "No directory selected.")
        return None  # User cancelled the selection
    
    # Ask if the user wants to create the directory if it doesn't exist
    if not os.path.exists(selected_dir):
        create = messagebox.askyesno("Directory does not exist", 
                                     f"The directory '{selected_dir}' does not exist. Do you want to create it?")
        if create:
            try:
                os.makedirs(selected_dir)
                messagebox.showinfo("Success", f"Directory '{selected_dir}' created successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create directory: {e}")
                return None
        else:
            return None  # User opted not to create the directory

    return selected_dir

# DO NOT DELETE BELOW CODE! to save time for testing, not using temporarly or using for test if commented out:
# file_path = choose_pth_file()
# if file_path:
#     print(f"Selected file: {file_path}")
# else:
#     print("No file was selected.")

# dir_path = select_or_create_directory()
# if dir_path:
#     print(f"Selected or created directory: {dir_path}")

# file_path="/project/hyperparam/hogefieldNum1_20241127_085000/pth/checkpoint_iter_5840.pth"
# file_path="/project/hyperparam/tomTest_20241129_162153/snn_model_tomTest.pth"
# file_path="/project/hyperparam/trial/pth/epoch_1.pth"
# # dir_path="/project/hyperparamTestTrainingv2"
# dir_path="/project/hyperparam/trial"
# dir_path="/project/hyperparam/trial/trial_hyp_change"
dir_path="/project/hyperparam/sat_less_timestep"

log_file="output.log"
log_path = os.path.join(dir_path, log_file)
plots_dir = os.path.join(dir_path,"plots")
pth_dir = os.path.join(dir_path,"pth")


# file_path = "hyperparam/trial/snn_model_hogefield.pth"
file_path = "/project/hyperparam/sat_less_timestep/snn_model_hogefield.pth"
# you need params in everything
chp_manager = CheckpointManager.load_checkpoint_with_defaults(file_path=file_path)
# chp_manager.print_contents()
params = chp_manager.get_hyperparameters()


#prprocess dataset by batch size
train_loader, test_loader = prepare_dataset(pth_file_path=file_path, params=params) # pth_file_path is decrepted in prpeare dataset func

loss_fn = SNNF.mse_count_loss(correct_rate=0.75, incorrect_rate=0.25,population_code=False)


#load model and states, if count is not equal to zero 
# checkpoint, epoch_at, loss_hist, acc_hist, counter = load_checkpoint(pth_file_path=file_path, model=model, optimizer=optimizer)


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

