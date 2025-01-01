import torch
from core.CheckPoinManager import CheckpointManager
import os
from utils.training_functional import prepare_dataset
####################################################################################
import numpy as np
from scipy.io import savemat
from sklearn.metrics import confusion_matrix
import os
import config.config as C

def export_generate_matlab(true_labels, predicted_labels, class_labels, test_acc_hist, output_dir="./", export_data=None, mat_file_name="metrics_results.mat"):
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

    # Save the MATLAB file with a dynamic name
    mat_file_path = os.path.join(output_dir, mat_file_name)
    savemat(mat_file_path, matlab_data)
    print(f"Data exported to {mat_file_path} for MATLAB.")





# List of 6 .pth files to compare
model_files = [
    "/project/hypertrain2/nb_n20/pth/epoch_64.pth",
    "/project/hypertrain2/nb_n40/pth/epoch_64.pth",
    "/project/hypertrain2/nb_n80/pth/epoch_64.pth",
    "/project/hypertrain2/wb_n20/pth/epoch_64.pth",
    "/project/hypertrain2/wb_n40/pth/epoch_64.pth",
    "/project/hypertrain2/wb_n80/pth/epoch_64.pth",
    # Add more model paths here as needed
]

# Common export directory
output_dir = C.DEFAULT_MATLAB_DIRECTORY

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all models
for i, model_file_path in enumerate(model_files):
    # Load Checkpoint
    checkpoint_manager = CheckpointManager.load_checkpoint_with_defaults_v1_1(model_file_path)
    params = checkpoint_manager.get_hyperparameters()

    # Prepare Dataset
    train_loader, test_loader = prepare_dataset(pth_file_path=model_file_path, params=params)

    # Evaluate Model
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

    checkpoint_manager.model.eval()
    with torch.no_grad():
        for data, targets, *_ in test_loader:
            data = data.permute(3, 0, 2, 1).squeeze().to(device)
            targets = targets.to(device)
            spk_rec = checkpoint_manager.model.forward(data, timestep_calculated)
            _, predicted = torch.max(spk_rec.sum(dim=0), 1)

            # # only number for class_names
            # true_labels.extend(targets.cpu().numpy())
            # predicted_labels.extend(predicted.cpu().numpy())
            
            # Map indices to class names
            true_labels.extend([class_name_mapping[idx] for idx in targets.cpu().numpy()])
            predicted_labels.extend([class_name_mapping[idx] for idx in predicted.cpu().numpy()])

    # Prepare hyperparameters for export
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

    # Generate the .mat file name dynamically
    mat_file_name = f"metrics_results_model_{i+1}.mat"

    # Call export_generate_matlab with unique file name
    export_generate_matlab(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_labels=sorted(set(true_labels + predicted_labels)),
        test_acc_hist=checkpoint_manager.get_test_acc_history(),
        output_dir=output_dir,
        export_data=export_data,
        mat_file_name=mat_file_name  # Egyedi fájlnév megadása
    )

print("Export for all models completed.")

