#load_model.py
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from core.Model import SNNModel  # Assuming the model is defined in model.py
import shutil
import os
import utils.loading_functional as FL
import utils.training_functional as FT
from torch.utils.data import DataLoader   
from snntorch import functional as SNNF 
import glob   
import matplotlib.pyplot as plt
import json
import re

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
        self.params = None
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
        if not file_path:
            self.status_label.config(text="Status: No file selected.")
            return

        try:
            # Load parameters from the JSON file
            with open(file_path, "r") as file:
                self.params = json.load(file)

            # Ensure the loaded parameters are consistent with the expected format
            self.params = {
                key.strip().lower(): value for key, value in self.params.items()
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
            # Translate file parameters to SNNModel parameters
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

            # Extract and standardize hyperparameters
            if "hyperparameters" not in checkpoint:
                raise ValueError("The .pth file does not contain hyperparameters.")
            
            self.params = {
                re.sub(r"[()\s]+", "_", key).strip("_").lower(): value
                for key, value in checkpoint["hyperparameters"].items()
            }

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

       
    def create_optimizer(self): 
        print(f"model params: {self.model.net.parameters()}")
        self.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=self.model.learning_rate, betas=(0.9, 0.999))
        self.loss_fn = SNNF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2,population_code=False, num_classes=34)
        
    def format_hyperparams(params):
        """
        Format the hyperparameters dictionary into a string for annotation.
        """
        return '\n'.join([f"{key}: {value}" for key, value in params.items()])
        
    def save_plot(fig, directory, filename):
        """
        Save a matplotlib figure to the specified directory with the given filename.

        Parameters:
        - fig (matplotlib.figure.Figure): The matplotlib figure to save.
        - directory (str): The directory where the plot will be saved.
        - filename (str): The filename of the plot (with extension, e.g., .png).

        Returns:
        None
        """
        filepath = os.path.join(directory, filename)
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")

    def save_checkpoint(checkpoint, checkpoint_dir, epoch=None, counter=None):
        """
        Save a training checkpoint with a dynamic name.

        Parameters:
        - checkpoint (dict): The checkpoint data to save.
        - checkpoint_dir (str): Directory where checkpoints are saved.
        - epoch (int, optional): The current epoch number. Used for naming.
        - counter (int, optional): The current iteration number. Used for naming.

        Raises:
        - ValueError: If neither `epoch` nor `counter` is provided.
        """
        # Ensure the directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Construct file name based on provided parameters
        if epoch is not None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        elif counter is not None:
            filename = f"checkpoint_iter_{counter}.pth"
        else:
            raise ValueError("Either 'epoch' or 'counter' must be provided for naming the checkpoint.")

        # Full path for the checkpoint
        filepath = os.path.join(checkpoint_dir, filename)

        # Save the checkpoint
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at: {filepath}")

    def save_checkpoint(checkpoint, checkpoint_dir, filename):
        """
        Save a checkpoint to the specified directory.

        Parameters:
        - checkpoint (dict): The checkpoint data to save.
        - checkpoint_dir (str): The directory to save the checkpoint in.
        - filename (str): The filename for the checkpoint.
        """
        # Ensure the checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Full path for the checkpoint file
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)

        print(f"Checkpoint saved to {checkpoint_path}")

    def delete_checkpoints(checkpoint_dir, pattern="checkpoint_iter_*.pth"):
        """
        Deletes checkpoint files matching a given pattern.

        Args:
            checkpoint_dir (str): Directory where checkpoint files are stored.
            pattern (str): Glob pattern to match checkpoint files. Default is "checkpoint_iter_*.pth".
        """
        files_to_delete = glob.glob(os.path.join(checkpoint_dir, pattern))
        for file_path in files_to_delete:
            os.remove(file_path)

    def start_training(self):
        """Start the training process."""
        try:
            # Get the number of epochs and iterations from the entry fields
            self.num_epochs = int(self.epochs_entry.get())
            self.num_iterations = int(self.iterations_entry.get())

            if self.num_epochs <= 0 or self.num_iterations <= 0:
                raise ValueError("Number of epochs and iterations must be greater than zero.")

            self.status_label.config(text=f"Status: Training for {self.num_epochs} epochs with {self.num_iterations} iterations each...")

            # Call the train function
            self.train()
            self.status_label.config(text="Status: Training completed!")
            messagebox.showinfo("Success", "Training completed successfully!")
        except ValueError as e:
            self.status_label.config(text="Status: Invalid input.")
            messagebox.showerror("Error", f"Invalid input: {e}")
        except Exception as e:
            self.status_label.config(text="Status: Training failed.")
            messagebox.showerror("Error", f"Training failed: {e}")
                                 
                                        
    def train(self):
        self.prepare_dataset()
        self.create_optimizer()
        # Initialize hyperparameters if not already defined
        start_epoch = 0
        loss_hist = []
        acc_hist = []
        test_acc_hist = []
        counter = 1
        save_every = 10  # Save checkpoint every 2 iterations
        plot_every = 100   # Plot and save the figure every 10 iterations
        
        checkpoint_dir=os.path.join(self.training_dir, 'p_checkpoints')
        plots_dir=os.path.join(self.training_dir, 'plots')
        # Create directories for checkpoints and plots if they don't exist
        os.makedirs(checkpoint_dir, 'p_checkpoints', exist_ok=True)
        os.makedirs(plots_dir, 'plots', exist_ok=True)
        
        # Find the latest checkpoint file in the 'checkpoints' folder
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pth')), key=os.path.getmtime)
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
        else:
            latest_checkpoint = None
            
        # Attempt to load the latest checkpoint
        if latest_checkpoint:
            try:
                checkpoint = torch.load(latest_checkpoint)
                self.model.net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                loss_hist = checkpoint['loss_hist']
                acc_hist = checkpoint['acc_hist']
                test_acc_hist = checkpoint.get('test_acc_hist', [])
                counter = checkpoint['counter']
                print(f"Resuming training from epoch {start_epoch}, iteration {counter}")
            except FileNotFoundError:
                print("No checkpoint found, starting training from scratch.")
        else:
            print("No checkpoint found, starting training from scratch.")    

        hyperparams_text = self.format_hyperparams(self.params)
        
        # Initialize the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        ax1.set_title("Training Loss")
        ax2.set_title("Training Accuracy")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Accuracy")

        # Adjust the figure layout to make room for the hyperparameter text
        fig.subplots_adjust(bottom=0.3)  # Adjust this value as needed  
        
        # Add the formatted hyperparameters text
        fig.text(0.5, 0.01, hyperparams_text, wrap=True, ha='center', fontsize=10)

        # Training loop
        for epoch in range(start_epoch, self.num_epochs):
            for i, batch in enumerate(iter(self.train_loader)):
                data, targets, *_ = batch
                data = data.permute(3, 0, 2, 1).squeeze()
                data = data.to(self.model.params.device)
                targets = targets.to(self.model.params.device)                 
        
                self.model.net.train()  # Set the network in training mode
                timesteps = self.params.get("timestep_calculated")                
                spk_rec = self.model.forward(self.model.net, data, timesteps)  # Forward pass
                loss_val = self.loss_fn(spk_rec, targets)

                # Gradient calculation + weight update
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()

                # Store loss and accuracy
                loss_hist.append(loss_val.item())
                acc = SNNF.accuracy_rate(spk_rec, targets)
                acc_hist.append(acc)
        
                # Update the plot and save the figure
                if counter % plot_every == 0:
                    ax1.clear()
                    ax2.clear()
                    ax1.set_title("Training Loss")
                    ax2.set_title("Training Accuracy")
                    ax1.set_xlabel("Iteration")
                    ax1.set_ylabel("Loss")
                    ax2.set_xlabel("Iteration")
                    ax2.set_ylabel("Accuracy")
                    ax1.plot(loss_hist, color='blue')
                    ax2.plot(acc_hist, color='red')
                    
                    # Save the epoch plot
                    plot_filename = f'training_plot_epoch_{epoch}_iter_{counter}.png'
                    self.save_plot(fig, plots_dir, plot_filename)

                print(f"\rEpoch {epoch}, Iteration {i} Train Loss: {loss_val.item():.2f} Accuracy: {acc*100:.2f}", end="")  
        
                if counter % save_every == 0:
                    # Save checkpoint every `save_every` iterations
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss_hist': loss_hist,
                        'acc_hist': acc_hist,
                        'test_acc_hist': test_acc_hist,
                        'counter': counter
                    }
            
                    checkpoint_filename = f'checkpoint_iter_{counter}.pth'
                    self.save_checkpoint(checkpoint, checkpoint_dir, checkpoint_filename)

                if counter % len(self.train_loader) == 0:
                    with torch.no_grad():
                        self.model.net.eval()
                        test_acc = FT.batch_accuracy(self.test_loader, self.model.net, self.params. timestep)
                        print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                        test_acc_hist.append(test_acc.item())
                counter += 1 

            # Save checkpoint after each epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss_hist': loss_hist,
                'acc_hist': acc_hist,
                'test_acc_hist': test_acc_hist,
                'counter': counter
            }
        
            # torch.save(checkpoint, f'p_checkpoints/checkpoint_epoch_{epoch}.pth')
            checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
            self.save_checkpoint(checkpoint, checkpoint_dir, checkpoint_filename)


            # Delete iteration checkpoints after saving the epoch checkpoint
            self.delete_checkpoints(checkpoint_dir)

            # Save the plot after each epoch
            plot_filename = f'training_plot_epoch_{epoch}.png'    
            self.save_plot(fig, plots_dir, plot_filename)
        
        # Save the final training plot
        final_plot_filename = 'final_training_plot.png'
        self.save_plot(fig, plots_dir, final_plot_filename)
        
        
        plt.close(fig)  # Close the plot to free up resources

        # Save the data to a JSON file
        training_data = {
            'loss_hist': loss_hist,
            'acc_hist': acc_hist,
            'test_acc_hist': test_acc_hist
        }
        with open('training_data.json', 'w') as f:
            json.dump(training_data, f)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
