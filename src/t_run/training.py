import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from src.core.Model import SNNModel  # Assuming the model is defined in model.py

import os

# Default directory for parameter files
DEFAULT_PARAM_DIR = "./hyperparam"
if not os.path.exists(DEFAULT_PARAM_DIR):  # Create the directory if it doesn't exist
    os.makedirs(DEFAULT_PARAM_DIR)

class TrainingApp:
    def __init__(self, parent):
        self.parent = parent
        self.parent.title("Training Application")

        # Create buttons for the app
        self.load_params_button = tk.Button(parent, text="Load Parameters", command=self.load_params)
        self.load_params_button.pack(padx=10, pady=5)

        self.create_model_button = tk.Button(parent, text="Create Model", command=self.create_model)
        self.create_model_button.pack(padx=10, pady=5)

        self.status_label = tk.Label(parent, text="Status: Waiting for input...")
        self.status_label.pack(padx=10, pady=10)

        # Store parameters and the model
        self.params = None
        self.model = None

    def load_params(self):
        """
        Open a file dialog to select a parameter file and load parameters.
        """
        file_path = filedialog.askopenfilename(
            initialdir=DEFAULT_PARAM_DIR,  # Open the default directory
            title="Select Parameter File",
            filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")),
        )
        if not file_path:
            self.status_label.config(text="Status: No file selected.")
            return

        try:
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Parse the file as key-value pairs
            self.params = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")  # Normalize key format
                    value = value.strip()
                    # Convert to int, float, or keep as string based on value content
                    if value.replace(".", "", 1).isdigit():
                        value = float(value) if "." in value else int(value)
                    self.params[key] = value

            self.status_label.config(text=f"Status: Parameters loaded from {file_path}")
            # Print the loaded parameters to the console
            print("Loaded Parameters:")
            for key, value in self.params.items():
                print(f"{key}: {value}")
            messagebox.showinfo("Success", "Parameters loaded successfully!")
        except Exception as e:
            self.status_label.config(text=f"Status: Failed to load parameters ({e})")
            messagebox.showerror("Error", f"Failed to load parameters: {e}")

    def create_model(self):
        """
        Create an instance of the SNNModel using loaded parameters.
        """
        if not self.params:
            self.status_label.config(text="Status: No parameters loaded.")
            messagebox.showwarning("Warning", "Please load parameters first.")
            return

        # Set default values for missing parameters
        defaults = {
            "num_inputs": 80,
            "num_hidden": 256,
            "num_outputs": 35,
            "beta_lif": 0.9,
            "treshold_lif": 0.5,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        # Mapping from file keys to SNNModel keys
        key_mapping = {
            "number_of_inputs": "num_inputs",
            "number_of_hidden_neurons": "num_hidden",
            "number_of_outputs": "num_outputs",
            "beta_(lif)": "beta_lif",
            "threshold_(lif)": "treshold_lif",
            "device": "device",
        }

        # Translate file parameters to SNNModel parameters
        params = {}
        for key, default in defaults.items():
            # Map the file key to the internal key
            file_key = [k for k, v in key_mapping.items() if v == key]
            if file_key and file_key[0] in self.params:
                params[key] = self.params[file_key[0]]
            else:
                params[key] = default  # Use default if key is missing

        try:
            # Print the parameters being used (for debugging)
            print("Model Parameters:")
            for key, value in params.items():
                print(f"{key}: {value}")

            # Create an instance of SNNModel
            self.model = SNNModel(
                num_inputs=params["num_inputs"],
                num_hidden=params["num_hidden"],
                num_outputs=params["num_outputs"],
                betaLIF=params["beta_lif"],
                tresholdLIF=params["treshold_lif"],
                device=params["device"],
            )
            self.status_label.config(text="Status: Model created successfully!")
            messagebox.showinfo("Success", "Model created successfully!")
        except Exception as e:
            self.status_label.config(text=f"Status: Failed to create model ({e})")
            messagebox.showerror("Error", f"Failed to create model: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
