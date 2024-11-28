import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import config.config as config
import utils.training_functional as TF
import subprocess
import pickle
import json
from datetime import datetime
from types import SimpleNamespace
import re
import torch
import shutil
from core.Model import SNNModel  # Import SNNModel dynamically

DEFAULT_DIR = "/project/hyperparam"
if not os.path.exists(DEFAULT_DIR):
    os.makedirs(DEFAULT_DIR)

model = None  # Global variable to store the SNNModel instance
params = {}  # Global variable to store hyperparameters

last_file_path = None  # Global variable to store the last file path

HYPERPARAMETERS = [
    ("Model ID", "hogefield", tk.StringVar), 
    ("Batch Size", config.BATCH_SIZE, tk.IntVar),
    ("SF Threshold", config.DEFAULT_THRESHOLD, tk.DoubleVar),
    ("Hop Length", config.DEFAULT_HOP_LENGTH, tk.IntVar),
    ("F Min", config.DEFAULT_F_MIN, tk.IntVar),
    ("F Max", config.DEFAULT_F_MAX, tk.IntVar),
    ("N Mels", config.DEFAULT_N_MELS, tk.IntVar),
    ("N FFT", config.DEFAULT_N_FFT, tk.IntVar),
    ("Wav File Samples", 16000, tk.IntVar),
    ("Timestep Calculated", TF.calculate_num_of_frames_constant(), tk.IntVar),
    ("Number of Inputs", config.NUMBER_OF_INPUTS_TO_NN, tk.IntVar),
    ("Number of Hidden Neurons", config.NUM_HIDDEN_NEURONS, tk.IntVar),
    ("Number of Outputs", config.NUMBER_OF_OUTPUTS_OF_NN, tk.IntVar),
    ("Beta LIF", config.BETA_LIF, tk.DoubleVar),
    ("Threshold LIF", config.THRESOLD_LIF, tk.DoubleVar),
    ("Device", config.DEVICE, tk.StringVar),
    ("Learning Rate", config.LEARNING_RATE, tk.DoubleVar),
    ("Filter Type", "custom", tk.StringVar, ["custom", "standard"]),
]
    
def get_hyperparameters():
    return HYPERPARAMETERS

# usage: batch_size_param = get_hyperparameter("Batch Size")


def create_model(h_params , model_id_value):
    """
    Create an instance of the SNNModel using loaded parameters and save it.
    """
    global model, last_file_path #,params
    
    if h_params is None:
        print("h_params is None")
    elif not h_params:
        print("h_params is empty")
    else:
        print("h_params content:", h_params)

    # Gather parameters from widgets
    # params = {label.lower().replace(" ", "_"): widget.get() for label, widget in hyperparameter_widgets}
    # model_id = model_id_var.get()
    save_dir = os.path.join(DEFAULT_DIR, model_id_value)
    if last_file_path:
        loaded_hyperparam_file = os.path.basename(last_file_path).replace(".json", "") #is like hogefield_20241128_185207,  
    else:
        messagebox.showerror("Error", "Load from or Save to a file the parameters to create directory where it should create the model.")
        return
    
    # Folder for saving the model based on the hyperparameter file
    save_dir = os.path.join(DEFAULT_DIR, loaded_hyperparam_file)

    # Confirm overwrite if the folder exists
    if os.path.exists(save_dir):
        if not messagebox.askyesno(
            "Overwrite Confirmation",
            f"The folder '{save_dir}' already exists. Do you want to clear it?"
        ):
            return
        else:
            shutil.rmtree(save_dir)
            print(f"Cleared folder: {save_dir}")

    os.makedirs(save_dir)
    print("cuda param is:",h_params.get("device", "cuda"))
    
    required_keys = [
        "number_of_inputs", "number_of_hidden_neurons", "number_of_outputs",
        "beta_lif", "threshold_lif", "device", "learning_rate"
    ]

    for key in required_keys:
        if key not in h_params:
            raise KeyError(f"Missing required parameter: '{key}'")

    # If all keys are present, proceed to construct `translated_params`
    translated_params = {
        "num_inputs": int(h_params["number_of_inputs"]),
        "num_hidden": int(h_params["number_of_hidden_neurons"]),
        "num_outputs": int(h_params["number_of_outputs"]),
        "betaLIF": float(h_params["beta_lif"]),
        "tresholdLIF": float(h_params["threshold_lif"]),
        "device": h_params["device"],
        "learning_rate": float(h_params["learning_rate"]),
    }
    
    
    try:
        # translated_params = {
        #     "num_inputs": int(h_params.get("number_of_inputs", 16)),
        #     "num_hidden": int(h_params.get("number_of_hidden_neurons", 256)),
        #     "num_outputs": int(h_params.get("number_of_outputs", 35)),
        #     "betaLIF": float(h_params.get("beta_lif", 0.9)),
        #     "tresholdLIF": float(h_params.get("threshold_lif", 0.5)),
        #     "device": h_params.get("device", "cuda"),
        #     "learning_rate": float(h_params.get("learning_rate", 0.0002)),
        # }
        # Create model instance
        # num_inputs, num_hidden, num_outputs, betaLIF, tresholdLIF, device, learning_rate
        print(h_params.get("device", "cuda"))
        model = SNNModel(**translated_params)
        
        print(h_params.get("device", "cuda"))
        
        # Save model
        model_path = os.path.join(save_dir, f"snn_model_{model_id_value}.pth")
        torch.save({"model_state": model.net.state_dict(), "hyperparameters": params}, model_path)
        messagebox.showinfo("Success", f"Model saved to {model_path}")
        print(f"Model saved to {model_path}")
                       
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create model: {e}")
        print(f"Error in create_model: {e}")



    # try:
    #     # Translate params for SNNModel
    #     translated_params = {
    #         "num_inputs": int(params.get("number_of_inputs", 16)),
    #         "num_hidden": int(params.get("number_of_hidden_neurons", 256)),
    #         "num_outputs": int(params.get("number_of_outputs", 35)),
    #         "betaLIF": float(params.get("beta_lif", 0.9)),
    #         "tresholdLIF": float(params.get("threshold_lif", 0.5)),
    #         "device": params.get("device", "cuda"),
    #         "learning_rate": float(params.get("learning_rate", 0.0002)),
    #     }

    #     # Create model instance
    #     from core.Model import SNNModel  # Import SNNModel dynamically
    #     model = SNNModel(**translated_params)

    #     # Save model
    #     model_path = os.path.join(save_dir, f"snn_model_{model_id}.pth")
    #     torch.save({"model_state": model.net.state_dict(), "hyperparameters": params}, model_path)
    #     messagebox.showinfo("Success", f"Model saved to {model_path}")
    #     print(f"Model saved to {model_path}")

    # except Exception as e:
    #     messagebox.showerror("Error", f"Failed to create model: {e}")
    #     print(f"Error in create_model: {e}")

def load_model():
    """
    Load parameters and model state from a .pth file.
    """
    global model, params
    file_path = filedialog.askopenfilename(
        initialdir=DEFAULT_DIR,
        title="Select a Model File",
        filetypes=(("PyTorch Model Files", "*.pth"), ("All Files", "*.*"))
    )

    if not file_path:
        return

    try:
        checkpoint = torch.load(file_path)

        if "hyperparameters" not in checkpoint:
            raise ValueError("The selected .pth file does not contain hyperparameters.")

        params = checkpoint["hyperparameters"]

        try:
            translated_params = {
                "num_inputs": int(params["number_of_inputs"]),
                "num_hidden": int(params["number_of_hidden_neurons"]),
                "num_outputs": int(params["number_of_outputs"]),
                "betaLIF": float(params["beta_lif"]),
                "tresholdLIF": float(params["threshold_lif"]),
                "device": params["device"],
                "learning_rate": float(params["learning_rate"]),
            }
        except KeyError as e:
            raise ValueError(f"Missing critical hyperparameter: {e}")


        from core.Model import SNNModel
        model = SNNModel(**translated_params)
        model.net.load_state_dict(checkpoint["model_state"])
        messagebox.showinfo("Success", f"Model loaded from {file_path}")
        print(f"Model loaded from {file_path}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")
        print(f"Error in load_model: {e}")

class WidgetManager:
    def __init__(self, parent, hyperparameters):
        """
        Initialize the WidgetManager with a list of hyperparameters and their associated widgets.
        """
        self.widgets = []
        self.frame = ttk.Frame(parent)
        self.frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Create widgets from hyperparameters
        for row, (label, default, var_type, *optional) in enumerate(HYPERPARAMETERS):
            var = var_type(value=default)
            choices = optional if optional else None  # Handle optional `choices`
            
            # Check if choices exist in *optional
            choices = optional[0] if optional else None
            if choices:
                # Create dropdown using ttk.Combobox
                widget = EntryWidget(self.frame, var, label_text=f"{label}:",default_value=default, choices=choices)
            else:
                # Create regular entry
                widget = EntryWidget(self.frame, var, label_text=f"{label}:", default_value=default)
            widget.grid(row=row, column=0, padx=5, pady=5, sticky="w")
            self.widgets.append((label, widget))
            
            # Initialize Filter Type logic if applicable
            self.init_filter_logic()

    def get_widget_by_label(self, label):
        """
        Retrieve a widget by its label.
        """
        for lbl, widget in self.widgets:
            if lbl == label:
                return widget
        return None

    def get_value_by_label(self, label):
        """
        Retrieve the value of a widget by its label.
        """
        widget = self.get_widget_by_label(label)
        if widget:
            return widget.get()
        return None
    
    def standardize_key(self, label):
        """
        Standardize a label by converting it to lowercase and replacing spaces with underscores.
        """
        return label.lower().replace(" ", "_")

    def get_hyperparameters(self):
        """
        Retrieve all hyperparameters as a dictionary.
        """
        return {self.standardize_key(label): widget.get() for label, widget in self.widgets}
    
    def init_filter_logic(self):
        """
        Add dynamic enable/disable logic for widgets based on 'Filter Type' selection.
        """
        filter_widget = self.get_widget_by_label("Filter Type")
        if not filter_widget:
            return  # If there's no "Filter Type", skip

        def update_field_states(*args):
            filter_type = filter_widget.get()
            for label, widget in self.widgets:
                if filter_type == "custom" and label in ["N Mels", "Number of Inputs", "F Min", "F Max"]:
                    widget.set_value({"N Mels": 16, "Number of Inputs": 16, "F Min": 300, "F Max": 8000}[label])
                    widget.entry.config(state="disabled")
                else:
                    widget.entry.config(state="normal")

        filter_widget.var.trace_add("write", lambda *args: update_field_states())  # Modern trace method
        update_field_states()  # Initialize field states    

class EntryWidget:
    def __init__(self, parent, variable, label_text="Enter:", default_value=10, choices=None, *args):
        self.var = variable
        self.choices = choices

        self.label = ttk.Label(parent, text=label_text)
        
        if self.choices:  # If choices are provided, create a Combobox
            self.entry = ttk.Combobox(parent, textvariable=self.var, values=self.choices, state="readonly")
            self.set_value(default_value)
        else:  # Otherwise, create a regular Entry
            self.entry = ttk.Entry(parent, textvariable=self.var)

    def grid(self, row, column, **kwargs):
        self.label.grid(row=row, column=column, **kwargs)
        self.entry.grid(row=row, column=column + 1, **kwargs)

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(value)

    def set_value(self, value):
        if self.choices:
            self.var.set(value)
        else:
            if isinstance(self.var, tk.StringVar):
                self.var.set(str(value))
            elif isinstance(self.var, (tk.IntVar, tk.DoubleVar)):
                self.var.set(value)
            else:
                raise TypeError("Unsupported variable type. Use StringVar, IntVar, or DoubleVar.")       


def open_file_and_load_data(hyperparameter_widgets):
    """
    Load hyperparameter values from a JSON file into the entry widgets.
    """
    global last_file_path
    file_path = filedialog.askopenfilename(
        initialdir=DEFAULT_DIR,
        title="Select a JSON File",
        filetypes=(("JSON Files", "*.json"), ("All Files", "*.*"))
    )
    if not file_path:
        return

    last_file_path = file_path
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        for label, widget in hyperparameter_widgets:
            key = label.lower().replace(" ", "_")
            if key in data:
                widget.set(data[key])

        messagebox.showinfo("Success", "Data loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read file: {str(e)}")

def save_parameters_to_json(file_path, hyperparameters):
    try:
        # Write to JSON file
        with open(file_path, "w") as file:
            json.dump(hyperparameters, file, indent=4)
        
        # Notify success
        messagebox.showinfo("Success", f"Parameters saved to {file_path}")
    except Exception as e:
        # Notify error
        messagebox.showerror("Error", f"Failed to save file: {str(e)}")


def get_save_file_path(default_filename, initial_dir=DEFAULT_DIR, title="Save Parameters"):
    """
    Prompt the user to select a file path for saving.

    Args:
    - default_filename (str): Suggested default filename.
    - initial_dir (str): Initial directory for the file dialog.
    - title (str): Title for the file dialog.

    Returns:
    - str: Selected file path or None if the user cancels.
    """
    file_path = filedialog.asksaveasfilename(
        initialdir=initial_dir,
        initialfile=default_filename,
        title=title,
        defaultextension=".json",
        filetypes=(("JSON Files", "*.json"), ("All Files", "*.*"))
    )
    return file_path if file_path else None

def save_parameters_to_file(hyperparameters, model_id_value):
    """
    Save the current hyperparameter values to a JSON file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"{model_id_value}_{timestamp}.json"
    # Prompt the user to select a file path for saving
    file_path = get_save_file_path(default_filename, initial_dir=DEFAULT_DIR, title="Save Parameters") 
    global last_file_path
    last_file_path=file_path
    # Convert widget values to parameters dictionary
    save_parameters_to_json(file_path, hyperparameters)


def save_to_last_file(hyperparameters, model_id_var=None):
    """
    Save hyperparameter values to the last loaded JSON file.
    """
    
    global last_file_path
    if not last_file_path:
        # If no last_file_path, prompt the user for a new file using model_id_var
        if model_id_var is None:
            messagebox.showerror("Error", "No previous file to save to, and no model ID provided.")
            return
        save_parameters_to_file(hyperparameters, model_id_var)
        return

    save_parameters_to_json(last_file_path, hyperparameters)


def create_hyperparams(hyperparameter_widgets):
    """
    Create a SimpleNamespace object containing hyperparameters.
    """
    hyperparams = {re.sub(r"\W|^(?=\d)", "_", label).lower(): widget.get() for label, widget in hyperparameter_widgets}
    return SimpleNamespace(**hyperparams)


def refresh_hyperparams(hyperparameter_widgets):
    """
    Refresh hyperparameters and notify an external app.
    """
    hyperparams = create_hyperparams(hyperparameter_widgets)

    with open("hyperparams.pkl", "wb") as file:
        pickle.dump(vars(hyperparams), file)

    with open("notification.txt", "w") as file:
        file.write("Data Updated")

    print("Hyperparameters refreshed and sent to the second app!")


def run_second_app():
    """
    Launch an external Python application.
    """
    subprocess.Popen(["python3", "/project/src/run/side_demonstrate_s2s.py"], start_new_session=True)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hyperparameter Manager")
    manager = WidgetManager(root, HYPERPARAMETERS)
    # # Example: Get specific hyperparameter
    # batch_size_widget = manager.get_widget_by_label("Batch Size")
    # if batch_size_widget:
    #     print(f"Batch Size: {batch_size_widget.get()}")

    ttk.Button(root, text="Save to File", command=lambda: save_parameters_to_file(manager.get_hyperparameters(), manager.get_value_by_label("Model ID") )).grid(
        row=2, column=0, padx=10, pady=5
    )
    ttk.Button(root, text="Load from File", command=lambda: open_file_and_load_data(hyperparameter_widgets)).grid(
        row=3, column=0, padx=10, pady=5
    )
    ttk.Button(root, text="Save to Last File", command=lambda: save_to_last_file(manager.get_hyperparameters()) ).grid(
        row=4, column=0, padx=10, pady=5
    )
    ttk.Button(root, text="Run Second App", command=run_second_app).grid(
        row=5, column=0, padx=10, pady=5)
    ttk.Button(root, text="Refresh Hyperparameters", command=lambda: refresh_hyperparams(hyperparameter_widgets)).grid(
        row=6, column=0, padx=10, pady=5
    )    
    ttk.Button(root, text="Create Model", command=lambda: create_model(manager.get_hyperparameters(), manager.get_value_by_label("Model ID") )).grid(
        row=7, column=0, padx=10, pady=5
    )
    ttk.Button(root, text="Load Model", command=load_model).grid(
        row=8, column=0, padx=10, pady=5
    )         

    root.mainloop()
