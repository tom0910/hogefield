import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from core.Model import SNNModel  # Assuming the model is defined in model.py
import shutil
import os
import json
import glob
from torch.utils.data import DataLoader   
from snntorch import functional as SNNF 
import matplotlib.pyplot as plt
import re
import utils.loading_functional as FL
from utils.training_utils import train

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
        
        self.load_pth_button = tk.Button(parent, text="Load Model Parameters (.pth)", command=self.load_pth_params)
        self.load_pth_button.pack(padx=10, pady=5)

        # Entry field for the number of epochs
        tk.Label(parent, text="Number of Epochs:").pack(padx=10, pady=5)
        self.epochs_entry = tk.Entry(parent)
        self.epochs_entry.insert(0, "10")  # Default value
        self.epochs_entry.pack(padx=10, pady=5)

        # Entry field for iterations per epoch
        tk.Label(parent, text="Iterations per Epoch:").pack(padx=10, pady=5)
        self.iterations_entry = tk.Entry(parent)
        self.iterations_entry.insert(0, "100")  # Default value
        self.iterations_entry.pack(padx=10, pady=5)

        # Start Training button
        self.start_training_button = tk.Button(parent, text="Start Training", command=self.start_training)
        self.start_training_button.pack(padx=10, pady=5)

        self.status_label = tk.Label(parent, text="Status: Waiting for input...")
        self.status_label.pack(padx=10, pady=10)

        # Store parameters and the model
        self.params = {}
        self.model = None
        self.param_entries = {}
        self.pth_file_path = None
        self.training_dir = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.loss_fn = None
        self.num_epochs = 10
        self.num_iterations = 100

    def load_params(self):
        """
        Open a file dialog to select a parameter JSON file and load parameters.
        """
        file_path = filedialog.askopenfilename(
            initialdir=DEFAULT_PARAM_DIR,  # Open the default directory
            title="Select Parameter File",
            filetypes=(("JSON Files", "*.json"), ("All Files", "*.*")),
        )
        
        if not file_path or not os.path.exists(file_path):
            self.status_label.config(text="Status: File not found.")
            messagebox.showerror("Error", "Selected .pth file does not exist.")
            return
                
        if not file_path:
            self.status_label.config(text="Status: No file selected.")
            return

        try:
            # Load parameters from the JSON file
            with open(file_path, "r") as file:
                self.params = json.load(file)
                print({key: type(value) for key, value in self.params.items()})
          
            self.params = {
                re.sub(r"_{2,}", "_", re.sub(r"[()\s]+", "_", key)).strip("_").lower(): value
                for key, value in self.params.items()
            }

            
            # Track the name of the loaded hyperparameter file (without extension)
            self.loaded_hyperparam_file = os.path.basename(file_path).replace(".json", "")

            self.status_label.config(text=f"Status: Parameters loaded from {file_path}")
            print("Loaded Parameters:")
            for key, value in self.params.items():
                print(f"{key}: {value}")

            # Display success message
            messagebox.showinfo("Success", "Parameters loaded successfully!")
        except json.JSONDecodeError:
            self.status_label.config(text="Status: Failed to parse JSON.")
            messagebox.showerror("Error", "The selected file is not a valid JSON file.")
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
            # Translate parameters to SNNModel parameters
            params = {
                "num_inputs": self.params["number_of_inputs"],
                "num_hidden": self.params["number_of_hidden_neurons"],
                "num_outputs": self.params["number_of_outputs"],
                "beta_lif": self.params["beta_lif"],  # Updated key
                "threshold_lif": self.params["threshold_lif"],  # Updated key
                "device": self.params["device"],
                "learning_rate": self.params["learning_rate"],
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
                tresholdLIF=params["threshold_lif"],
                device=params["device"],
                learning_rate=params["learning_rate"],
            )

            # Save the model
            model_path = os.path.join(save_dir, f"snn_model_{self.params.get('model_id', 'default')}.pth")
            torch.save({"model_state": self.model.net.state_dict(), "hyperparameters": self.params}, model_path)
            self.status_label.config(text=f"Status: Model saved to {model_path}")
            messagebox.showinfo("Success", f"Model saved successfully to {model_path}")
            print(f"Model saved to {model_path}")
            self.pth_file_path = model_path
            self.training_dir = model_path
        except Exception as e:
            self.status_label.config(text="Status: Failed to create model.")
            messagebox.showerror("Error", f"Failed to create model: {e}")
    def load_pth_params(self):
        """
        Load parameters from a .pth file and populate the UI fields.
        """
        file_path = filedialog.askopenfilename(
            initialdir=DEFAULT_PARAM_DIR,
            title="Select a Model File",
            filetypes=(("PyTorch Model Files", "*.pth"), ("All Files", "*.*"))
        )

        if not file_path or not os.path.exists(file_path):
            self.status_label.config(text="Status: No file selected or file does not exist.")
            messagebox.showerror("Error", "No valid .pth file selected.")
            return

        try:
            # Load the .pth file
            checkpoint = torch.load(file_path)

            # Extract and standardize hyperparameters
            if "hyperparameters" not in checkpoint:
                raise ValueError("The selected .pth file does not contain hyperparameters.")
            
            self.params = checkpoint["hyperparameters"]

            # Normalize keys to match UI field names
            self.params = {
                re.sub(r"[()\s]+", "_", key).strip("_").lower(): value
                for key, value in self.params.items()
            }

            self.pth_file_path = file_path  # Save the loaded file path
            self.populate_param_fields()  # Populate the UI
            self.status_label.config(text=f"Parameters loaded from {file_path}")
            messagebox.showinfo("Success", "Parameters loaded successfully!")
        except Exception as e:
            self.status_label.config(text=f"Failed to load parameters: {e}")
            messagebox.showerror("Error", f"Could not load .pth file: {e}")


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
            # Disable specific widgets if filter_type is "custom"
            filter_type = self.params.get("filter_type", "").lower()
            if filter_type == "custom":
                for key in ["f_min", "f_max", "n_mels"]:
                    if key in self.param_entries:
                        self.param_entries[key].config(state="disabled")
            else:
                for key in ["f_min", "f_max", "n_mels"]:
                    if key in self.param_entries:
                        self.param_entries[key].config(state="normal")            
             
    def start_training(self):
        """
        Start the training process.
        """
        try:
            # Validate if parameters are loaded
            if not self.params:
                self.status_label.config(text="Status: No parameters loaded.")
                messagebox.showerror("Error", "Please load parameters before starting training.")
                return

            # Validate the number of epochs and iterations
            self.num_epochs = int(self.epochs_entry.get())
            self.num_iterations = int(self.iterations_entry.get())
            if self.num_epochs <= 0 or self.num_iterations <= 0:
                raise ValueError("Number of epochs and iterations must be greater than zero.")

            # Prepare dataset
            self.prepare_dataset()

            # Create optimizer and loss function
            self.create_optimizer()

            # Initialize model
            if not self.model:
                self.create_model()
                
                
            params = FL.load_parameters_from_pth(model_path)   
            # Align keys with SNNModel requirements
            paramsSNN = {
                "num_inputs":       params.get("number_of_inputs", 16),
                "num_hidden":       params.get("number_of_hidden_neurons", 256),
                "num_outputs":      params.get("number_of_outputs", 35),
                "beta_lif":         params.get("beta_lif", 0.9),  # Default if missing
                "threshold_lif":    params.get("threshold_lif", 0.5),  # Default if missing
                "device":           params.get("device", "cuda"),
                "learning_rate":    params.get("learning_rate", 0.0002),
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
                learning_rate=params.get("learning_rate", 0.0002),
                num_classes=35,
            )
            # Start training
            # Train the model
            train(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                test_loader=test_loader,
                params=params,
                loss_fn=loss_fn,
                num_epochs=self.num_epochs,
                checkpoint_dir=checkpoint_dir,
                plots_dir=plots_dir,
            )

            # Update status on completion
            self.status_label.config(text="Status: Training completed!")
            messagebox.showinfo("Success", "Training completed successfully!")
        except ValueError as e:
            self.status_label.config(text=f"Status: Invalid input: {e}")
            messagebox.showerror("Error", f"Invalid input: {e}")
        except Exception as e:
            self.status_label.config(text=f"Status: Training failed: {e}")
            messagebox.showerror("Error", f"Training failed: {e}")

    def prepare_dataset(self):
        """
        Prepare training and test datasets.
        """
        if not self.params:
            raise ValueError("Hyperparameters are required to prepare the dataset.")

        # Assuming `prepare_datasets` is a helper function
        train_set, test_set, target_labels = FL.prepare_datasets(
            data_path="/project/data/GSC/", check_labels=False
        )

        batch_size = int(self.params.get("batch_size", 32))  # Default to 32 if not found

        # Create DataLoader for training and testing
        self.train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            collate_fn=lambda batch: FL.custom_collate_fn(batch, self.params, target_labels)
        )
        self.test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: FL.custom_collate_fn(batch, self.params, target_labels)
        )

    def create_optimizer(self):
        """
        Create an optimizer and loss function.
        """
        if not self.model:
            raise ValueError("Model must be created before initializing optimizer.")

        learning_rate = float(self.params.get("learning_rate", 0.001))  # Default to 0.001

        self.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=learning_rate)
        self.loss_fn = SNNF.mse_count_loss(
            correct_rate=0.8, incorrect_rate=0.2, population_code=False,
            num_classes=self.params.get("num_outputs", 10)  # Default to 10 classes
        )
        
            

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
