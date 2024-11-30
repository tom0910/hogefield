import torch


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

    def save(self, file_path):
        """
        Save the checkpoint to a file.
        """
        checkpoint = {
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
        }
        torch.save(checkpoint, file_path)

    @staticmethod
    def load_checkpoint_with_defaults(file_path):
        """
        Load a checkpoint from a file and initialize a CheckpointManager with the loaded data.

        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            CheckpointManager: An instance of CheckpointManager with loaded data, including model and optimizer (if present).
        """
        checkpoint = torch.load(file_path)

        # Initialize model and optimizer variables
        model = None
        optimizer = None

        # Load model if available
        if "model_state_dict" in checkpoint:
            from core.Model import SNNModel  # Adjust based on your code structure
            hyperparameters = checkpoint.get("hyperparameters", {})
            if all(key in hyperparameters for key in ["number_of_inputs", "number_of_hidden_neurons", "number_of_outputs", "beta_lif", "threshold_lif", "device"]):
                model = SNNModel(
                    num_inputs=hyperparameters["number_of_inputs"],
                    num_hidden=hyperparameters["number_of_hidden_neurons"],
                    num_outputs=hyperparameters["number_of_outputs"],
                    betaLIF=hyperparameters["beta_lif"],
                    tresholdLIF=hyperparameters["threshold_lif"],
                    device=hyperparameters["device"],
                )
                model.net.load_state_dict(checkpoint["model_state_dict"])
            else:
                print("Warning: Hyperparameters required to initialize the model are missing in the checkpoint.")

        # Load optimizer if available
        optimizer_type = checkpoint.get("optimizer_type")
        optimizer_params = checkpoint.get("optimizer_params", {})
        # learning_rate = optimizer_params.get("lr", hyperparameters.get("learning_rate", 0.001))  # Default to 0.001 if missing
        if "optimizer_state_dict" in checkpoint and model:
            optimizer_map = {
                "Adam": torch.optim.Adam,
                "SGD": torch.optim.SGD,
                "RMSprop": torch.optim.RMSprop,
            }
            optimizer_class = optimizer_map.get(optimizer_type, torch.optim.Adam)
            optimizer = optimizer_class(model.net.parameters(), **optimizer_params)
            # optimizer = optimizer_class(model.net.parameters(), lr=learning_rate)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Extract additional data
        loss_hist = checkpoint.get("loss_hist", [])
        acc_hist = checkpoint.get("acc_hist", [])
        test_acc_hist = checkpoint.get("test_acc_hist", [])
        counter = checkpoint.get("counter", 0)
        epoch = checkpoint.get("epoch", 0)
        hyperparameters = checkpoint.get("hyperparameters", {})

        # Create and return a CheckpointManager with loaded data
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
        print("Loss History:", self.loss_hist)
        print("Accuracy History:", self.acc_hist)
        print("Test Accuracy History:", self.test_acc_hist)
        print("Counter:", self.counter)
        print("Counter:", self.epoch)
        print("===================================")





# class CheckpointManager:
    
#     def __init__(self, model=None, optimizer=None, hyperparameters=None):
#         """
#         Initialize the CheckpointManager with model, optimizer, hyperparameters, and additional data.

#         Args:
#             model (torch.nn.Module): Model instance.
#             optimizer (torch.optim.Optimizer): Optimizer instance.
#             hyperparameters (dict): Model hyperparameters.
#             additional_data (dict): Additional data like loss history, epoch, etc.
#         """
#         self.model = model
#         self.optimizer = optimizer
#         self.hyperparameters = hyperparameters or {}
#         self.loss_hist = []
#         self.acc_hist = []
#         self.test_acc_hist = []
#         self.counter = 0
#         self.optimizer_type = type(optimizer).__name__ if optimizer else None
#         self.optimizer_params = {"lr": optimizer.param_groups[0]["lr"]} if optimizer else {}

#     def save(self, file_path):
#         """
#         Save the checkpoint to a file.
#         """
#         checkpoint = {
#             "model_state_dict": self.model.net.state_dict() if self.model else None,
#             "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
#             "optimizer_type": self.optimizer_type,
#             "optimizer_params": self.optimizer_params,
#             "hyperparameters": self.hyperparameters,
#             "loss_hist": self.loss_hist,
#             "acc_hist": self.acc_hist,
#             "test_acc_hist": self.test_acc_hist,
#             "counter": self.counter,
#         }
#         torch.save(checkpoint, file_path)


#     @staticmethod
#     def load_checkpoint(file_path, model=None, optimizer=None):
#         """
#         Load a checkpoint from a file.

#         Args:
#             file_path (str): Path to the checkpoint file.
#             model (torch.nn.Module): Optional model instance to load the weights into.
#             optimizer (torch.optim.Optimizer): Optional optimizer instance to load the state into.

#         Returns:
#             CheckpointManager: A new or updated instance of CheckpointManager.
#         """
#         checkpoint = torch.load(file_path)

#         # Load model state if available
#         if "model_state_dict" in checkpoint and model:
#             model.net.load_state_dict(checkpoint["model_state_dict"])

#         # Load optimizer state if available
#         if "optimizer_state_dict" in checkpoint and optimizer:
#             optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#         # Extract stored data
#         hyperparameters = checkpoint.get("hyperparameters", {})
#         loss_hist = checkpoint.get("loss_hist", [])
#         acc_hist = checkpoint.get("acc_hist", [])
#         test_acc_hist = checkpoint.get("test_acc_hist", [])
#         counter = checkpoint.get("counter", 0)

#         # Handle optimizer type and params
#         optimizer_type = checkpoint.get("optimizer_type")
#         optimizer_params = checkpoint.get("optimizer_params", {})

#         # Dynamically create optimizer if none exists
#         if optimizer_type and model:
#             optimizer_map = {
#                 "Adam": torch.optim.Adam,
#                 "SGD": torch.optim.SGD,
#                 "RMSprop": torch.optim.RMSprop,
#             }
#             optimizer_class = optimizer_map.get(optimizer_type, torch.optim.Adam)
#             optimizer = optimizer_class(model.net.parameters(), **optimizer_params)

#         # Create and return a new CheckpointManager instance
#         manager = CheckpointManager(
#             model=model,
#             optimizer=optimizer,
#             hyperparameters=hyperparameters,
#         )
#         manager.loss_hist = loss_hist
#         manager.acc_hist = acc_hist
#         manager.test_acc_hist = test_acc_hist
#         manager.counter = counter

#         return manager
      

#     def get_hyperparameters(self):
#         """
#         Retrieve the stored hyperparameters.

#         Returns:
#             dict: The hyperparameters dictionary.
#         """
#         return self.hyperparameters

#     def set_hyperparameters(self, hyperparameters):
#         """
#         Update the hyperparameters.

#         Args:
#             hyperparameters (dict): The new hyperparameters.
#         """
#         self.hyperparameters.update(hyperparameters)

