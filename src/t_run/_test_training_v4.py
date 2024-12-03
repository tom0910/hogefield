import os
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from utils.training_functional import prepare_dataset
from core.Model import SNNModel
from core.CheckPoinManager import CheckpointManager
from utils.training_utils import train_as_hypp_change
from snntorch import functional as SNNF

def main_train_v1_1():
    # Select file and directory
    # file_path, dir_path = select_pth_and_dir()
    # if not file_path or not dir_path:
    #     print("File or directory not selected. Exiting.")
    #     return

    # file_path="/project/hyperparam/checkLIF_099_01_20241202_221419/snn_model_checkLIF_099_01.pth"
    # dir_path="/project/hyperparam/checkLIF_099_01_20241202_221419"
    
    # file_path ="/project/hyperparam/checkLIF_095_02_20241202_223240/snn_model_checkLIF_095_02.pth"
    # dir_path  ="/project/hyperparam/checkLIF_095_02_20241202_223240"
    
    # file_path = "/project/hyperparam/checkLIF_09_02_20241202_231742/snn_model_checkLIF_09_02.pth"
    # dir_path  = "/project/hyperparam/checkLIF_09_02_20241202_231742"
    
    file_path = "/project/hyperparam/checkLIF_09_02_20241202_231742/snn_model_checkLIF_09_02.pth"
    dir_path  = "/project/hyperparam/checkLIF_09_02_20241202_231742"

    # Setup directories
    log_file = "output.log"
    log_path = os.path.join(dir_path, log_file)
    plots_dir = os.path.join(dir_path, "plots")
    pth_dir = os.path.join(dir_path, "pth")

    # Load checkpoint
    chp_manager = CheckpointManager.load_checkpoint_with_defaults_v1_1(file_path=file_path)  # Use the v1.1 function
    params = chp_manager.get_hyperparameters()

    # Dynamically determine model type
    model_type = params.get("model_type", "SNNModel")  # Default to SNNModel if not specified

    # Align keys for the model class
    model_hyperparams = {
        "num_inputs": params.get("number_of_inputs", 16),
        "num_hidden": params.get("number_of_hidden_neurons", 256),
        "num_outputs": params.get("number_of_outputs", 35),
        "betaLIF": params.get("beta_lif", 0.9),
        "tresholdLIF": params.get("threshold_lif", 0.5),
        "device": params.get("device", "cuda"),
    }

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
    
    
main_train_v1_1()