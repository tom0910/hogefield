import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from src.core.Model import SNNModel  # Assuming the model is defined in model.py
import shutil
import os

# Default directory for parameter files
DEFAULT_PARAM_DIR = "./hyperparam"
if not os.path.exists(DEFAULT_PARAM_DIR):  # Create the directory if it doesn't exist
    os.makedirs(DEFAULT_PARAM_DIR)

class TrainingApp:
    def __init__(self, parent):
        self.parent = parent
        self.parent.title("Training Application")
        self.param_entries = {}  # Dictionary to store parameter entry fields


        # Create buttons for the app
        self.load_params_button = tk.Button(parent, text="Load Parameters", command=self.load_params)
        self.load_params_button.pack(padx=10, pady=5)

        self.create_model_button = tk.Button(parent, text="Create Model", command=self.create_model)
        self.create_model_button.pack(padx=10, pady=5)

        self.show_params_button = tk.Button(parent, text="Show Parameters", command=self.populate_param_fields)
        self.show_params_button.pack(padx=10, pady=5)

        self.status_label = tk.Label(parent, text="Status: Waiting for input...")
        self.status_label.pack(padx=10, pady=10)
        
        self.load_pth_button = tk.Button(parent, text="Load Model Parameters (.pth)", command=self.load_pth_params)
        self.load_pth_button.pack(padx=10, pady=5)


        # Store parameters and the model
        self.params = None
        self.model = None
        
        # Store parameters and the model
        self.params = None
        self.model = None
        self.param_entries = {}

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

            # Track the name of the loaded hyperparameter file (without extension)
            self.loaded_hyperparam_file = os.path.basename(file_path).replace(".txt", "")

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

        if not hasattr(self, "loaded_hyperparam_file") or not self.loaded_hyperparam_file:
            self.status_label.config(text="Status: Missing hyperparameter file name.")
            messagebox.showerror("Error", "Failed to identify the hyperparameter file name. Please reload parameters.")
            return

        # Folder for saving the model based on the hyperparameter file
        save_dir = os.path.join(DEFAULT_PARAM_DIR, self.loaded_hyperparam_file)

        # Ask for confirmation if folder exists
        if os.path.exists(save_dir):
            if not messagebox.askyesno(
                "Overwrite Confirmation",
                f"The folder '{save_dir}' already exists. Do you want to clear it?"
            ):
                self.status_label.config(text="Status: Model creation cancelled.")
                return
            else:
                shutil.rmtree(save_dir)  # Clear the folder
                print(f"Cleared folder: {save_dir}")

        os.makedirs(save_dir)  # Create folder

        try:
            # Translate file parameters to SNNModel parameters
            params = {
                "num_inputs": self.params["number_of_inputs"],
                "num_hidden": self.params["number_of_hidden_neurons"],
                "num_outputs": self.params["number_of_outputs"],
                "beta_lif": self.params["beta_(lif)"],
                "treshold_lif": self.params["threshold_(lif)"],
                "device": self.params["device"],
            }

            # Print the parameters being used
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

            # Save the model
            model_path = os.path.join(save_dir, f"snn_model_{self.params.get('model_id', 'default')}.pth")
            torch.save({"model_state": self.model.net.state_dict(), "hyperparameters": self.params}, model_path)
            self.status_label.config(text=f"Status: Model saved to {model_path}")
            messagebox.showinfo("Success", f"Model saved successfully to {model_path}")
            print(f"Model saved to {model_path}")
        except Exception as e:
            self.status_label.config(text="Status: Failed to create model.")
            messagebox.showerror("Error", f"Failed to create model: {e}")
        
    def populate_param_fields(self):
        """
        Populate the parameter fields in the UI with the current parameters.
        """
        if not self.params:
            messagebox.showwarning("Warning", "No parameters loaded.")
            return

        # Create a new frame for parameter fields
        if hasattr(self, "param_frame"):
            self.param_frame.destroy()  # Clear the old frame if it exists
        self.param_frame = tk.Frame(self.parent)
        self.param_frame.pack(padx=10, pady=10)

        # Dynamically create labels and entry fields for each parameter
        for idx, (key, value) in enumerate(self.params.items()):
            label = tk.Label(self.param_frame, text=key.replace("_", " ").capitalize() + ":")
            label.grid(row=idx, column=0, sticky="w", padx=5, pady=5)

            entry = tk.Entry(self.param_frame, width=20)
            entry.grid(row=idx, column=1, sticky="w", padx=5, pady=5)
            entry.insert(0, str(value))  # Pre-fill with the current value

            self.param_entries[key] = entry  # Save the entry widget for later use 
    
    def load_pth_params(self):
        """
        Load parameters from a .pth file and display them in the UI.
        """
        file_path = filedialog.askopenfilename(
            initialdir=DEFAULT_PARAM_DIR,
            title="Select a Model File",
            filetypes=(("PyTorch Model Files", "*.pth"), ("All Files", "*.*"))
        )

        if not file_path:
            self.status_label.config(text="Status: No file selected.")
            return

        try:
            # Load the .pth file
            checkpoint = torch.load(file_path)

            # Extract hyperparameters
            if "hyperparameters" not in checkpoint:
                raise ValueError("The .pth file does not contain hyperparameters.")

            self.params = checkpoint["hyperparameters"]

            # Track the loaded model name
            self.loaded_hyperparam_file = os.path.basename(file_path).replace(".pth", "")

            # Print parameters to the console
            print("Loaded Parameters from .pth:")
            for key, value in self.params.items():
                print(f"{key}: {value}")

            # Populate parameters in the UI
            self.populate_param_fields()
            self.status_label.config(text=f"Status: Parameters loaded from {file_path}")
            messagebox.showinfo("Success", "Parameters loaded successfully!")
        except Exception as e:
            self.status_label.config(text=f"Status: Failed to load parameters ({e})")
            messagebox.showerror("Error", f"Failed to load parameters: {e}")
           



if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
