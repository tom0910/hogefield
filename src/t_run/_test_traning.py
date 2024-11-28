#_test_training.py
from utils.training_utils import train
from utils.training_functional import prepare_dataset, create_optimizer
from core.Model import SNNModel
import utils.loading_functional as FL

import sys
import os


base_dir="/project/hyperparam/hogefieldNum1_20241127_085000/"
log_file="output.log"
log_path = os.path.join(base_dir, log_file)
plots_dir = os.path.join(base_dir,"plots")
pth_dir = os.path.join(base_dir,"pth")
model_path = os.path.join(base_dir,"snn_model_default.pth")

# Redirect print statements to a file
with open(log_file, "w") as log_file:
    sys.stdout = log_file
    print("This will go to the file!")
    
# Reset stdout back to default
sys.stdout = sys.__stdout__
print("This will appear in the terminal.")

# Load parameters from the .pth file
model_path = model_path
params = FL.load_parameters_from_pth(model_path)  # Updated function ensures normalized keys

# Align keys with SNNModel requirements
paramsSNN = {
    "num_inputs":       params.get("number_of_inputs", 16),
    "num_hidden":       params.get("number_of_hidden_neurons", 256),
    "num_outputs":      params.get("number_of_outputs", 35),
    "beta_lif":         params.get("beta_lif", 0.9),  # Default if missing
    "threshold_lif":    params.get("threshold_lif", 0.5),  # Default if missing
    "device":           params.get("device", "cuda"),
    "learning_rate":    params.get("learning_rate", 0.0002),
}

# Prepare dataset
train_loader, test_loader = prepare_dataset(pth_file_path=model_path, params=params)

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
    learning_rate=params.get("learning_rate", 0.0002),
    num_classes=35,
)

# Training configuration
num_epochs = 100
checkpoint_dir = pth_dir
plots_dir = plots_dir

# Train the model
train(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    test_loader=test_loader,
    params=params,
    loss_fn=loss_fn,
    num_epochs=num_epochs,
    checkpoint_dir=checkpoint_dir,
    plots_dir=plots_dir,
)
