import unittest
import torch
from torch import nn
# from your_script import load_optimizer  # Replace with the actual module name

def load_optimizer(model, checkpoint_path, default_optimizer="Adam", default_params={"lr": 0.05}):
    """
    Load an optimizer from a checkpoint, with a fallback to a default optimizer if metadata is missing.
    
    Args:
        model (torch.nn.Module): The model whose parameters the optimizer is managing.
        checkpoint_path (str): Path to the checkpoint file.
        default_optimizer (str): Default optimizer type if metadata is missing. Default is 'Adam'.
        default_params (dict): Default parameters for the fallback optimizer.
    
    Returns:
        optimizer (torch.optim.Optimizer): The loaded or newly created optimizer.
    """
    checkpoint = torch.load(checkpoint_path)

    # Check for optimizer metadata
    optimizer_type = checkpoint.get("optimizer_type", default_optimizer)
    optimizer_params = checkpoint.get("optimizer_params", default_params)

    # Map optimizer types to their constructors
    optimizer_map = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop,
    }

    # Select the optimizer class
    optimizer_class = optimizer_map.get(optimizer_type, torch.optim.Adam)

    # Create the optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("Warning: No optimizer state dict found in checkpoint. Using default settings.")
        optimizer_params = {"lr": 0.001}  # Ensure fallback params are correctly aligned
        optimizer = optimizer_class(model.parameters(), **optimizer_params)


    return optimizer


class TestLoadOptimizer(unittest.TestCase):
    def setUp(self):
        # Sample model
        self.model = nn.Sequential(nn.Linear(10, 10))
        self.checkpoint_path = "test_checkpoint.pth"

    def tearDown(self):
        # Clean up test file
        import os
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def test_load_with_metadata(self):
        # Save checkpoint with metadata
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_type": "Adam",
            "optimizer_params": {"lr": 0.01}
        }
        torch.save(checkpoint, self.checkpoint_path)

        # Load optimizer
        loaded_optimizer = load_optimizer(self.model, self.checkpoint_path)
        self.assertIsInstance(loaded_optimizer, torch.optim.Adam)
        self.assertAlmostEqual(loaded_optimizer.param_groups[0]["lr"], 0.01)

    def test_fallback_to_default_optimizer(self):
        # Save checkpoint without optimizer metadata
        checkpoint = {
            "model_state_dict": self.model.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_path)

        # Load optimizer
        loaded_optimizer = load_optimizer(self.model, self.checkpoint_path)
        self.assertIsInstance(loaded_optimizer, torch.optim.Adam)
        self.assertAlmostEqual(loaded_optimizer.param_groups[0]["lr"], 0.001)

    def test_missing_state_dict(self):
        # Save checkpoint with metadata but no state_dict
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_type": "SGD",
            "optimizer_params": {"lr": 0.05}
        }
        torch.save(checkpoint, self.checkpoint_path)

        # Load optimizer
        loaded_optimizer = load_optimizer(self.model, self.checkpoint_path)
        self.assertIsInstance(loaded_optimizer, torch.optim.SGD)
        self.assertAlmostEqual(loaded_optimizer.param_groups[0]["lr"], 0.05)

    def test_unsupported_optimizer(self):
        # Save checkpoint with unsupported optimizer type
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_type": "UnknownOptimizer",
            "optimizer_params": {"lr": 0.01}
        }
        torch.save(checkpoint, self.checkpoint_path)

        # Load optimizer
        loaded_optimizer = load_optimizer(self.model, self.checkpoint_path)
        self.assertIsInstance(loaded_optimizer, torch.optim.Adam)
        self.assertAlmostEqual(loaded_optimizer.param_groups[0]["lr"], 0.001)

if __name__ == "__main__":
    unittest.main()


# python3 -m unittest _test_optimizer_check.py 