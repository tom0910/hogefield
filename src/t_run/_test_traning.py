#_test_training.py
from utils.training_utils import train
from utils.training_functional import prepare_dataset, create_optimizer
from core.Model import SNNModel
import utils.loading_functional as FL

# Load parameters from the .pth file
model_path = "/project/hyperparam/second_20241124_202729/snn_model_default.pth"
params = FL.load_parameters_from_pth(model_path)  # Updated function used here

# Align keys with SNNModel requirements
paramsSNN = {
    "num_inputs":       params["number_of_inputs"],
    "num_hidden":       params["number_of_hidden_neurons"],
    "num_outputs":      params["number_of_outputs"],
    "beta_lif":         params["beta_lif"],  # Updated standardized key
    "threshold_lif":    params["threshold_lif"],  # Updated standardized key
    "device":           params["device"],
    "learning_rate":    params["learning_rate"],
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
    learning_rate=params["learning_rate"],
    num_classes=35,
)

# Training configuration
num_epochs = 100
checkpoint_dir = "/project/hyperparam/second_20241124_202729/checkpoints"
plots_dir = "/project/hyperparam/second_20241124_202729/plots"

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


# from utils.training_utils import train
# from utils.training_functional import prepare_dataset,create_optimizer
# from core.Model import SNNModel
# import utils.loading_functional as FL 

# # Setup2
# model_path = "/project/hyperparam/second_20241124_202729/snn_model_default.pth"
# params = FL.load_parameters_from_pth(model_path)
# paramsSNN = {
#     "num_inputs":       params["number_of_inputs"],
#     "num_hidden":       params["number_of_hidden_neurons"],
#     "num_outputs":      params["number_of_outputs"],
#     "beta_lif":         params["beta_(lif)"],
#     "treshold_lif":     params["threshold_(lif)"],
#     "device":           params["device"],
#     "learning_rate":    params["learning_rate"],
# }
# train_loader,test_loader= prepare_dataset(pth_file_path=model_path, params=params)
# model = SNNModel(
#     num_inputs      = paramsSNN["num_inputs"],
#     num_hidden      = paramsSNN["num_hidden"],
#     num_outputs     = paramsSNN["num_outputs"],
#     betaLIF         = paramsSNN["beta_lif"],
#     tresholdLIF     = paramsSNN["treshold_lif"],
#     device          = paramsSNN["device"],
#     learning_rate   = paramsSNN["learning_rate"],
# )
# optimizer, loss_fn = create_optimizer(net_params=model.net.parameters(), learning_rate=params["learning_rate"],num_classes=35)
# num_epochs = 100

# # Paths
# checkpoint_dir   = "/project/hyperparam/second_20241124_202729/checkpoints"
# plots_dir        = "/project/hyperparam/second_20241124_202729/plots"

# train(
#     model=model,
#     optimizer=optimizer,
#     train_loader=train_loader,
#     test_loader=test_loader,
#     params=params,
#     loss_fn=loss_fn,
#     num_epochs=num_epochs,
#     checkpoint_dir=checkpoint_dir,
#     plots_dir=plots_dir,
# )
