from run.side_maintrain import main_train_v1_1
from core.CheckPoinManager import CheckpointManager

pth_saved_dir   = "/project/hypertrain/nband_nmel16_4lyr_rd_synaptic_512hn"
pth_save_path   = "/project/hypertrain/nband_nmel16_4lyr_rd_synaptic_512hn/snn_model_hogefield.pth"

dir_path = pth_saved_dir
file_path = pth_save_path

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


main_train_v1_1(pth_saved_dir, pth_save_path)