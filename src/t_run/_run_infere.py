import torch
from core.CheckPoinManager import CheckpointManager
import os



# Define the path to your saved model
model_file_path = "/project/hypertrain/wband_nmel80_4lyr/pth/epoch_84.pth"


# Extract the base file name without extension
base_name = os.path.splitext(model_file_path)[0]

# Load the checkpoint using CheckpointManager
checkpoint_manager = CheckpointManager.load_checkpoint_with_defaults_v1_1(model_file_path)
checkpoint_manager.update_batch_size(100)
test_acc_hist = checkpoint_manager.get_test_acc_history()
# checkpoint_manager.save(new_file_path)

# Print confirmation
print("Model successfully loaded!")

from utils.training_functional import prepare_dataset

# Prepare train and test loaders
params = checkpoint_manager.get_hyperparameters()
train_loader, test_loader = prepare_dataset(pth_file_path=model_file_path, params=params) # file path here does not matter, function does not use that

print("DataLoaders prepared successfully!")

# HYPERPARAMETERS = [
#     ("Model ID", "hogefield", tk.StringVar), 
#     ("Batch Size", config.BATCH_SIZE, tk.IntVar),
#     ("SF Threshold", config.DEFAULT_THRESHOLD, tk.DoubleVar),
#     ("Hop Length", config.DEFAULT_HOP_LENGTH, tk.IntVar),
#     ("F Min", config.DEFAULT_F_MIN, tk.IntVar),
#     ("F Max", config.DEFAULT_F_MAX, tk.IntVar),
#     ("N Mels", config.DEFAULT_N_MELS, tk.IntVar),
#     ("N FFT", config.DEFAULT_N_FFT, tk.IntVar),
#     ("Wav File Samples", 16000, tk.IntVar),
#     ("Timestep Calculated", TF.calculate_num_of_frames_constant(), tk.IntVar),
#     ("Number of Inputs", config.NUMBER_OF_INPUTS_TO_NN, tk.IntVar),
#     ("Number of Hidden Neurons", config.NUM_HIDDEN_NEURONS, tk.IntVar),
#     ("Number of Outputs", config.NUMBER_OF_OUTPUTS_OF_NN, tk.IntVar),
#     ("Beta LIF", config.BETA_LIF, tk.DoubleVar),
#     ("Threshold LIF", config.THRESOLD_LIF, tk.DoubleVar),
#     ("Device", config.DEVICE, tk.StringVar),
#     ("Learning Rate", config.LEARNING_RATE, tk.DoubleVar),
#     ("Filter Type", "custom", tk.StringVar, ["custom", "standard", "narrowband"]),
#     ("Model Type", "SNNModel", tk.StringVar, ["SNNModel", "SNNModel_population", "SNNModel_droput", "DynamicSNNModel", "RD_SNNModel", "RD_SNNModel_Synaptic"]),  # New entry for v1.1
#     ("Correct Rate", 1, tk.DoubleVar),  
#     ("Incorrect Rate", 0, tk.DoubleVar),  
# ]

export_data = {
    "num_inputs": params.get("number_of_inputs"),
    "num_hidden": params.get("number_of_hidden_neurons"),
    "num_outputs": params.get("number_of_outputs"),
    "betaLIF": params.get("beta_lif"),
    "thresholdLIF": params.get("threshold_lif"),
    "learning_rate": params.get("learning_rate"),
    "filter_type": params.get("filter_type"),
    "n_mels": params.get("n_mels"),
    "model_type": params.get("model_type", "SNNModel"),
    "hop_length": params.get("hop_length"),
    "f_max": params.get("f_max"),
    "f_min": params.get("f_min"),
    "file_path": model_file_path
}


import torch

true_labels = []
predicted_labels = []
device = params.get("device")
timestep_calculated = params.get("timestep_calculated")

class_name_mapping = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
    'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
    'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
    'visual', 'wow', 'yes', 'zero'
]

# Set the model to evaluation mode
checkpoint_manager.model.eval()

# Evaluate the model on the test set
with torch.no_grad():
    for data, targets, *_ in test_loader:
        data = data.permute(3, 0, 2, 1).squeeze().to(device)
        targets = targets.to(device)

        # Get model predictions
        spk_rec = checkpoint_manager.model.forward(data, timestep_calculated)
        _, predicted = torch.max(spk_rec.sum(dim=0), 1)

        # Map indices to class names
        true_labels.extend([class_name_mapping[idx] for idx in targets.cpu().numpy()])
        predicted_labels.extend([class_name_mapping[idx] for idx in predicted.cpu().numpy()])


print("Model evaluation completed!")

####################################################################################
import numpy as np
from scipy.io import savemat
from sklearn.metrics import confusion_matrix
import os

# Define the function to export and generate MATLAB script
def export_generate_matlab(true_labels, predicted_labels, class_labels, test_acc_hist, output_dir="./", export_data=None):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_labels, normalize="true")

    # Generate performance metrics
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = np.sum(cm) - (fp + fn + tp)

    # Calculate metrics
    precision = tp / (tp + fp + 1e-10)  # Precision = TP / (TP + FP)
    sensitivity = tp / (tp + fn + 1e-10)  # Sensitivity (Recall) = TP / (TP + FN)
    specificity = tn / (tn + fp + 1e-10)  # Specificity = TN / (TN + FP)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)  # F1-Score

    # Prepare data for MATLAB export
    matlab_data = {
        "class_labels": class_labels,
        "confusion_matrix": cm,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score,
        "test_acc_hist": test_acc_hist
    }

    # Merge export_data into matlab_data if provided
    if export_data:
        matlab_data.update(export_data)

    # Save the MATLAB file
    mat_file_path = os.path.join(output_dir, "metrics_results.mat")
    savemat(mat_file_path, matlab_data)
    print(f"Data exported to {mat_file_path} for MATLAB.")

# Example usage (replace with actual inference labels and test accuracy history):
# true_labels = ["cat", "dog", "bird", "cat"]  # Replace with your inference data
# predicted_labels = ["cat", "dog", "cat", "bird"]  # Replace with your inference data
class_labels = sorted(set(true_labels + predicted_labels))

# Call the export function
export_generate_matlab(true_labels, predicted_labels, class_labels, test_acc_hist, output_dir="./",export_data=export_data)
