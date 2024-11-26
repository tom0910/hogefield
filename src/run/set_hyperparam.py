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

DEFAULT_DIR = "/project/hyperparam"
if not os.path.exists(DEFAULT_DIR):
    os.makedirs(DEFAULT_DIR)

last_file_path = None  # Global variable to store the last file path

class EntryWidget:
    def __init__(self, parent, variable, label_text="Enter:", default_value=10, choices=None):
        self.var = variable
        self.choices = choices

        self.label = ttk.Label(parent, text=label_text)
        if choices:
            self.entry = ttk.Combobox(parent, textvariable=self.var, values=choices, state="readonly")
            self.set_value(default_value)
        else:
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



# class EntryWidget:
#     def __init__(self, parent, variable, label_text="Enter:", default_value=10):
#         self.var = variable
#         self.set_value(default_value)

#         self.label = ttk.Label(parent, text=label_text)
#         self.entry = ttk.Entry(parent, textvariable=self.var)

#     def grid(self, row, column, **kwargs):
#         self.label.grid(row=row, column=column, **kwargs)
#         self.entry.grid(row=row, column=column + 1, **kwargs)

#     def get(self):
#         return self.var.get()

#     def set(self, value):
#         self.var.set(value)

#     def set_value(self, value):
#         if isinstance(self.var, tk.StringVar):
#             self.var.set(str(value))
#         elif isinstance(self.var, (tk.IntVar, tk.DoubleVar)):
#             self.var.set(value)
#         else:
#             raise TypeError("Unsupported variable type. Use StringVar, IntVar, or DoubleVar.")

def create_hyperparameter_fields(parent):
    """
    Create and layout hyperparameter fields with labels and default values.
    """
    model_id_var = tk.StringVar(value="hogefield")
    model_id_widget = EntryWidget(parent, model_id_var, label_text="Model ID:", default_value="hogefield")
    model_id_widget.grid(0, 0, padx=5, pady=5, sticky="w")

    hyperparameters = [
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
        ("Beta (LIF)", config.BETA_LIF, tk.DoubleVar),
        ("Threshold (LIF)", config.THRESOLD_LIF, tk.DoubleVar),
        ("Device", config.DEVICE, tk.StringVar),
        ("Learning Rate", config.LEARNING_RATE, tk.DoubleVar),
    ]

    frame = ttk.Frame(parent)
    frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    widgets = []
    for row, (label, default, var_type) in enumerate(hyperparameters):
        var = var_type(value=default)
        widget = EntryWidget(frame, var, label_text=f"{label}:", default_value=default)
        widget.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        widgets.append((label, widget))

    # Add Filter Type as a choice field
    filter_type_var = tk.StringVar(value="custom")
    filter_type_widget = EntryWidget(
        frame,
        filter_type_var,
        label_text="Filter Type:",
        default_value="custom",
        choices=["custom", "standard"],
    )
    filter_type_widget.grid(row=len(hyperparameters), column=0, padx=5, pady=5, sticky="w")

    # Add logic for dynamic enable/disable
    def update_field_states(*args):
        filter_type = filter_type_var.get()
        for label, widget in widgets:
            if filter_type == "custom" and label in ["N Mels", "F Min", "F Max"]:
                widget.set_value({"N Mels": 16, "F Min": 300, "F Max": 8000}[label])
                widget.entry.config(state="disabled")
            else:
                widget.entry.config(state="normal")

        if filter_type == "custom":
            # Number of Inputs should remain editable
            for label, widget in widgets:
                if label == "Number of Inputs":
                    widget.entry.config(state="normal")

    filter_type_var.trace("w", update_field_states)
    update_field_states()  # Initialize field states

    widgets.append(("Filter Type", filter_type_widget))

    return widgets, model_id_var


# def create_hyperparameter_fields(parent):
#     """
#     Create and layout hyperparameter fields with labels and default values.
#     """
#     model_id_var = tk.StringVar(value="hogefield")
#     model_id_widget = EntryWidget(parent, model_id_var, label_text="Model ID:", default_value="hogefield")
#     model_id_widget.grid(0, 0, padx=5, pady=5, sticky="w")

#     hyperparameters = [
#         ("Batch Size", config.BATCH_SIZE, tk.IntVar),
#         ("SF Threshold", config.DEFAULT_THRESHOLD, tk.DoubleVar),
#         ("Hop Length", config.DEFAULT_HOP_LENGTH, tk.IntVar),
#         ("F Min", config.DEFAULT_F_MIN, tk.IntVar),
#         ("F Max", config.DEFAULT_F_MAX, tk.IntVar),
#         ("N Mels", config.DEFAULT_N_MELS, tk.IntVar),
#         ("N FFT", config.DEFAULT_N_FFT, tk.IntVar),
#         ("Wav File Samples", 16000, tk.IntVar),
#         ("Timestep Calculated", TF.calculate_num_of_frames_constant(), tk.IntVar),
#         ("Filter Type Custom or Standard", "custom", tk.StringVar),
#         ("Number of Inputs", config.NUMBER_OF_INPUTS_TO_NN, tk.IntVar),
#         ("Number of Hidden Neurons", config.NUM_HIDDEN_NEURONS, tk.IntVar),
#         ("Number of Outputs", config.NUMBER_OF_OUTPUTS_OF_NN, tk.IntVar),
#         ("Beta (LIF)", config.BETA_LIF, tk.DoubleVar),
#         ("Threshold (LIF)", config.THRESOLD_LIF, tk.DoubleVar),
#         ("Device", config.DEVICE, tk.StringVar),
#         ("Learning Rate", config.LEARNING_RATE, tk.DoubleVar),
#     ]

#     frame = ttk.Frame(parent)
#     frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

#     widgets = []
#     for row, (label, default, var_type) in enumerate(hyperparameters):
#         var = var_type(value=default)
#         widget = EntryWidget(frame, var, label_text=f"{label}:", default_value=default)
#         widget.grid(row=row, column=0, padx=5, pady=5, sticky="w")
#         widgets.append((label, widget))

#     return widgets, model_id_var


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


def save_parameters_to_file(hyperparameter_widgets, model_id_var):
    """
    Save the current hyperparameter values to a JSON file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"{model_id_var.get()}_{timestamp}.json"

    file_path = filedialog.asksaveasfilename(
        initialdir=DEFAULT_DIR,
        initialfile=default_filename,
        title="Save Parameters",
        defaultextension=".json",
        filetypes=(("JSON Files", "*.json"), ("All Files", "*.*"))
    )
    if not file_path:
        return

    try:
        parameters = {label.lower().replace(" ", "_"): widget.get() for label, widget in hyperparameter_widgets}
        with open(file_path, "w") as file:
            json.dump(parameters, file, indent=4)

        messagebox.showinfo("Success", f"Parameters saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save file: {str(e)}")


def save_to_last_file(hyperparameter_widgets):
    """
    Save hyperparameter values to the last loaded JSON file.
    """
    global last_file_path
    if not last_file_path:
        save_parameters_to_file(hyperparameter_widgets)
        return

    try:
        parameters = {label.lower().replace(" ", "_"): widget.get() for label, widget in hyperparameter_widgets}
        with open(last_file_path, "w") as file:
            json.dump(parameters, file, indent=4)

        messagebox.showinfo("Success", f"Parameters saved to {last_file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save file: {str(e)}")


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

    hyperparameter_widgets, model_id_var = create_hyperparameter_fields(root)

    ttk.Button(root, text="Save to File", command=lambda: save_parameters_to_file(hyperparameter_widgets, model_id_var)).grid(
        row=2, column=0, padx=10, pady=5
    )
    ttk.Button(root, text="Load from File", command=lambda: open_file_and_load_data(hyperparameter_widgets)).grid(
        row=3, column=0, padx=10, pady=5
    )
    ttk.Button(root, text="Save to Last File", command=lambda: save_to_last_file(hyperparameter_widgets)).grid(
        row=4, column=0, padx=10, pady=5
    )
    ttk.Button(root, text="Run Second App", command=run_second_app).grid(row=5, column=0, padx=10, pady=5)
    ttk.Button(root, text="Refresh Hyperparameters", command=lambda: refresh_hyperparams(hyperparameter_widgets)).grid(
        row=6, column=0, padx=10, pady=5
    )

    root.mainloop()
