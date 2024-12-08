import torch
from core.Model import SNNModel, SNNModel_population, SNNModel_droput, DynamicSNNModel


class CheckpointManager:
    def __init__(self, model=None, optimizer=None, hyperparameters=None):
        """
        Initialize the CheckpointManager with model, optimizer, hyperparameters, and additional data.

        Args:
            model (torch.nn.Module): Model instance.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            hyperparameters (dict): Model hyperparameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.hyperparameters = hyperparameters or {}
        self.loss_hist = []
        self.acc_hist = []
        self.test_acc_hist = []
        self.epoch = 0
        self.counter = 0
        self.optimizer_type = type(optimizer).__name__ if optimizer else None
        self.optimizer_params = {"lr": optimizer.param_groups[0]["lr"]} if optimizer else {}
        # self.correct_rate = 1
        # self.incorrect_rate = 0

    def save(self, file_path):
        """
        Save the checkpoint to a file.
        """
        checkpoint = {
            "model_type": type(self.model).__name__,  # Save model type fro v1.1 handling new models
            "model_state_dict": self.model.net.state_dict() if self.model else None,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "optimizer_type": self.optimizer_type,
            "optimizer_params": self.optimizer_params,
            "hyperparameters": self.hyperparameters,
            "loss_hist": self.loss_hist,
            "acc_hist": self.acc_hist,
            "test_acc_hist": self.test_acc_hist,
            "epoch": self.epoch,
            "counter": self.counter,
            # "correct_rate":self.correct_rate,
            # "incorrect_rate":self.incorrect_rate,
        }
        torch.save(checkpoint, file_path)

    MODEL_REGISTRY = {
        "SNNModel": SNNModel,
        "SNNModel_population": SNNModel_population,
        "SNNModel_droput": SNNModel_droput,
        "DynamicSNNModel" : DynamicSNNModel
        
    }
    
    @staticmethod
    def load_checkpoint_with_defaults_v1_1(file_path):
        checkpoint = torch.load(file_path)

        # Extract model type and hyperparameters
        model_type = checkpoint.get("model_type", "SNNModel")  # Default to SNNModel 
        hyperparameters = checkpoint.get("hyperparameters", {})

        # Dynamically instantiate the model
        if model_type in CheckpointManager.MODEL_REGISTRY:
            model_class = CheckpointManager.MODEL_REGISTRY[model_type]
            model = model_class(
                num_inputs=hyperparameters.get("number_of_inputs"),
                num_hidden=hyperparameters.get("number_of_hidden_neurons"),
                num_outputs=hyperparameters.get("number_of_outputs"),
                betaLIF=hyperparameters.get("beta_lif"),
                tresholdLIF=hyperparameters.get("threshold_lif"),
                device=hyperparameters.get("device"),
            )
            model.net.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Initialize optimizer
        optimizer = None
        optimizer_type = checkpoint.get("optimizer_type")
        optimizer_params = checkpoint.get("optimizer_params", {})
        if "optimizer_state_dict" in checkpoint and model:
            optimizer_map = {
                "Adam": torch.optim.Adam,
                "SGD": torch.optim.SGD,
                "RMSprop": torch.optim.RMSprop,
            }
            optimizer_class = optimizer_map.get(optimizer_type, torch.optim.Adam)
            optimizer = optimizer_class(model.net.parameters(), **optimizer_params)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Extract additional data
        manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            hyperparameters=hyperparameters,
        )
        manager.loss_hist = checkpoint.get("loss_hist", [])
        manager.acc_hist = checkpoint.get("acc_hist", [])
        manager.test_acc_hist = checkpoint.get("test_acc_hist", [])
        manager.counter = checkpoint.get("counter", 0)
        manager.epoch = checkpoint.get("epoch", 0)
        # manager.correct_rate = checkpoint.get("correct_rate",1)
        # manager.incorrect_rate = checkpoint.get("incorrect_rate",0)
        correct_rate    = hyperparameters["correct_rate"]
        incorrect_rate  = hyperparameters["incorrect_rate"]
        return manager


    @staticmethod
    def load_checkpoint_with_model(file_path, model=None, optimizer=None):
        """
        Load a checkpoint from a file.

        Args:
            file_path (str): Path to the checkpoint file.
            model (torch.nn.Module): Optional model instance to load the weights into.
            optimizer (torch.optim.Optimizer): Optional optimizer instance to load the state into.

        Returns:
            CheckpointManager: A new or updated instance of CheckpointManager.
        """
        checkpoint = torch.load(file_path)

        # Load model state if available
        if "model_state_dict" in checkpoint and model:
            model.net.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if available
        if "optimizer_state_dict" in checkpoint and optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Extract stored data
        hyperparameters = checkpoint.get("hyperparameters", {})
        loss_hist = checkpoint.get("loss_hist", [])
        acc_hist = checkpoint.get("acc_hist", [])
        test_acc_hist = checkpoint.get("test_acc_hist", [])
        counter = checkpoint.get("counter", 0)
        epoch = checkpoint.get("epoch", 0)
        ## change these:
        # correct_rate = checkpoint.get("correct_rate",1)
        # incorrect_rate = checkpoint.get("incorrect_rate",0)
        correct_rate    = hyperparameters["correct_rate"]
        incorrect_rate  = hyperparameters["incorrect_rate"]

        # Handle optimizer type and params
        optimizer_type = checkpoint.get("optimizer_type")
        optimizer_params = checkpoint.get("optimizer_params", {})

        # Dynamically create optimizer if none exists
        if optimizer_type and model:
            optimizer_map = {
                "Adam": torch.optim.Adam,
                "SGD": torch.optim.SGD,
                "RMSprop": torch.optim.RMSprop,
            }
            optimizer_class = optimizer_map.get(optimizer_type, torch.optim.Adam)
            optimizer = optimizer_class(model.net.parameters(), **optimizer_params)

        # Create and return a new CheckpointManager instance
        manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            hyperparameters=hyperparameters,
        )
        manager.loss_hist = loss_hist
        manager.acc_hist = acc_hist
        manager.test_acc_hist = test_acc_hist
        manager.counter = counter
        manager.epoch = epoch
        manager.correct_rate = correct_rate
        manager.incorrect_rate = incorrect_rate

        return manager

    def initialize_model_and_optimizer(self):
        """
        Initialize the model and optimizer using stored hyperparameters and optimizer settings.

        Returns:
            tuple: (SNNModel, Optimizer)
        """
        # Validate hyperparameters for model creation
        required_keys = ["num_inputs", "num_hidden", "num_outputs", "betaLIF", "tresholdLIF", "device"]
        missing_keys = [key for key in required_keys if key not in self.hyperparameters]
        if missing_keys:
            raise KeyError(f"Missing required hyperparameters: {', '.join(missing_keys)}")

        # Dynamically create the model
        from core.Model import SNNModel  # Adjust import path if needed
        self.model = SNNModel(
            num_inputs=self.hyperparameters["num_inputs"],
            num_hidden=self.hyperparameters["num_hidden"],
            num_outputs=self.hyperparameters["num_outputs"],
            betaLIF=self.hyperparameters["betaLIF"],
            tresholdLIF=self.hyperparameters["tresholdLIF"],
            device=self.hyperparameters["device"],
        )

        # Dynamically create the optimizer
        optimizer_map = {
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD,
            "RMSprop": torch.optim.RMSprop,
        }
        optimizer_class = optimizer_map.get(self.optimizer_type, torch.optim.Adam)
        self.optimizer = optimizer_class(self.model.net.parameters(), **self.optimizer_params)

        return self.model, self.optimizer

    def get_hyperparameters(self):
        """
        Retrieve the stored hyperparameters.

        Returns:
            dict: The hyperparameters dictionary.
        """
        return self.hyperparameters

    # not implemented
    def set_hyperparameters(self, hyperparameters):
        """
        Update the hyperparameters.

        Args:
            hyperparameters (dict): The new hyperparameters.
        """
        self.hyperparameters.update(hyperparameters)
        
    def parse_counter(filename):
        """
        Parse the counter from the checkpoint filename.
        
        Args:
            filename (str): The checkpoint filename to parse.
            
        Returns:
            int: Extracted counter value, or 0 if not valid.
        """
        filename_parts = filename.split('/')[-1].split('.')[0].split('_')
        if len(filename_parts) > 1 and filename_parts[-1].isdigit():
            return int(filename_parts[-1])
        return 0

    def print_contents(self):
        """
        Print the current state and contents of the CheckpointManager instance to the terminal.
        """
        print("=== CheckpointManager Contents ===")
        print("Model:", "Present" if self.model else "None")
        print("Optimizer:", "Present" if self.optimizer else "None")
        print("Optimizer Type:", self.optimizer_type)
        print("Optimizer Parameters:", self.optimizer_params)
        print("Hyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f"  {key}: {value}")
        print("Loss History (Last 5):", self.loss_hist[-5:] if self.loss_hist else "No Data")
        print("Accuracy History (Last 5):", self.acc_hist[-5:] if self.acc_hist else "No Data")
        print("Test Accuracy History (Last 5):", self.test_acc_hist[-5:] if self.test_acc_hist else "No Data")
        print("Counter:", self.counter)
        print("Epoch:", self.epoch)
        print("correct rate:", self.correct_rate)
        print("incorrect rate:", self.incorrect_rate)
        print("===================================")
