from tkinter import filedialog, Tk, messagebox, simpledialog
import os
import tkinter as tk
from utils.training_functional import prepare_dataset, create_optimizer, load_latest_checkpoint, load_checkpoint
import utils.loading_functional as FL
from core.Model import SNNModel
from utils.training_utils import train, train_as_hypp_change

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
file_path="/project/hyperparam/tomTest_20241129_162153/snn_model_tomTest.pth"
# # dir_path="/project/hyperparamTestTrainingv2"
dir_path="/project/hyperparam/trial"

log_file="output.log"
log_path = os.path.join(dir_path, log_file)
plots_dir = os.path.join(dir_path,"plots")
pth_dir = os.path.join(dir_path,"pth")
# model_path = os.path.join(file_path,"snn_model_default.pth")

# Prepare dataset
params = FL.load_parameters_from_pth(file_path)  # Updated function ensures normalized keys
# check params for debug
print(params)
#prprocess dataset by batch size
train_loader, test_loader = prepare_dataset(pth_file_path=file_path, params=params) # pth_file_path is decrepted in prpeare dataset func

# Align keys with SNNModel requirements
paramsSNN = {
    "num_inputs":       params.get("number_of_inputs"),
    "num_hidden":       params.get("number_of_hidden_neurons"),
    "num_outputs":      params.get("number_of_outputs"),
    "beta_lif":         params.get("beta_lif"),  # Default if missing
    "threshold_lif":    params.get("threshold_lif"),  # Default if missing
    "device":           params.get("device"),
    "learning_rate":    params.get("learning_rate"),
}

# Initialize model
model = SNNModel(
    num_inputs      = paramsSNN["num_inputs"],
    num_hidden      = paramsSNN["num_hidden"],
    num_outputs     = paramsSNN["num_outputs"],
    betaLIF         = paramsSNN["beta_lif"],
    tresholdLIF     = paramsSNN["threshold_lif"],  # Fixed typo from "treshold"
    device          = paramsSNN["device"],
    learning_rate   = paramsSNN["learning_rate"],
)


# Create optimizer and loss function
optimizer, loss_fn = create_optimizer(
    net_params=model.net.parameters(),
    learning_rate=params.get("learning_rate"),
    num_classes=params.get("num_outputs"),
)

#load model and states, if count is not equal to zero 
# checkpoint, epoch_at, loss_hist, acc_hist, counter = load_checkpoint(pth_file_path=file_path, model=model, optimizer=optimizer)


# Train the model
train_as_hypp_change(
    file_path,
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    test_loader=test_loader,
    params=params,
    loss_fn=loss_fn,
    num_epochs=100,
    checkpoint_dir=pth_dir,
    plots_dir=plots_dir,
)

