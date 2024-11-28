import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import os
import config.config as config
import utils.training_functional as TF
import subprocess
import pickle 

DEFAULT_DIR = "/project/hyperparam"
if not os.path.exists(DEFAULT_DIR):  # Create the folder if it doesn't exist
    os.makedirs(DEFAULT_DIR)

last_file_path = None  # Global variable to store the last file path


class EntryWidget:
    def __init__(self, parent, variable, label_text="Enter:", default_value=10):
        self.var = variable
        if isinstance(self.var, tk.StringVar):
            self.var.set(str(default_value))
        elif isinstance(self.var, (tk.IntVar, tk.DoubleVar)):
            self.var.set(default_value)
        else:
            raise TypeError("Unsupported variable type: Use StringVar, IntVar, or DoubleVar.")
        
        self.label = ttk.Label(parent, text=label_text)
        self.entry = ttk.Entry(parent, textvariable=self.var)

    def grid(self, row, column, **kwargs):
        self.label.grid(row=row, column=column, **kwargs)
        self.entry.grid(row=row, column=column + 1, **kwargs)

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(value)

def create_hyperparameter_fields(parent):
    """
    Create a systematic column of entry fields for the hyperparameters.
    """
    # Create a field for Model ID with a default value
    model_id_var = tk.StringVar(value="hogefield")
    model_id_widget = EntryWidget(parent, model_id_var, label_text="Model ID:", default_value="hogefield")

    # Place the Model ID widget in its own row in the parent grid
    model_id_widget.label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
    model_id_widget.entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")


    # Hyperparameters and their default values
    hyperparameters = [
        ("Batch Size", config.BATCH_SIZE, tk.IntVar),
        ("SF Threshold", config.DEFAULT_THRESHOLD, tk.DoubleVar),
        ("Hop Length", config.DEFAULT_HOP_LENGTH, tk.IntVar),
        ("F Min", config.DEFAULT_F_MIN, tk.IntVar),
        ("F Max", config.DEFAULT_F_MAX, tk.IntVar),
        ("N Mels", config.DEFAULT_N_MELS, tk.IntVar),
        ("N FFT", config.DEFAULT_N_FFT, tk.IntVar),
        ("Wav File Samples", 16000, tk.IntVar),
        ("Timestep Calculated", TF.calculate_num_of_frames_constant(), tk.IntVar), # should be calculated
        ("Filter Type custom or standard", "custom", tk.StringVar),
        
        # SNNModel-specific parameters
        # ("Number of Inputs", TF.calculate_number_of_input_neurons(), tk.IntVar), # should be calculated
        ("Number of Inputs", config.NUMBER_OF_INPUTS_TO_NN, tk.IntVar),
        ("Number of Hidden Neurons", config.NUM_HIDDEN_NEURONS, tk.IntVar),
        ("Number of Outputs", config.NUMBER_OF_OUTPUTS_OF_NN, tk.IntVar),
        ("Beta (LIF)", config.BETA_LIF, tk.DoubleVar),
        ("Threshold (LIF)", config.THRESOLD_LIF, tk.DoubleVar),
        ("Device", config.DEVICE, tk.StringVar),
        ("learning_rate", config.LEARNING_RATE, tk.DoubleVar),
    ]

    # Create a frame for the hyperparameters
    frame = ttk.Frame(parent)
    frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")  # Put frame below Model ID

    # Create EntryWidgets for each hyperparameter
    widgets = []
    for row, (label, default, var_type) in enumerate(hyperparameters):
        var = var_type(value=default)
        widget = EntryWidget(frame, variable=var, label_text=f"{label}:", default_value=default)
        widget.grid(row=row, column=0, padx=5, pady=5, sticky="w")  # Align label and entry in frame
        widgets.append((label, widget))

    return widgets, model_id_var

# # Access the values from the widgets
# for label, widget in hyperparameter_widgets:
#     if label == "Number of Inputs":
#         num_inputs = widget.get()
#     elif label == "Beta (LIF)":
#         beta = widget.get()
# # And so on...



def open_file_and_load_data(hyperparameter_widgets):
    """
    Open a file manager to select a text file and load its data into the entry fields.
    Assumes the file contains 'key: value' pairs.
    """
    global last_file_path  # Track the last opened file path
    file_path = filedialog.askopenfilename(
        initialdir=DEFAULT_DIR,
        title="Select a Text File",
        filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
    )
    if file_path:
        last_file_path = file_path  # Save the file path for future use


    if not file_path:
        return  # User cancelled the dialog

    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Ensure we have enough lines to fill all fields
        if len(lines) != len(hyperparameter_widgets):
            messagebox.showerror(
                "File Error",
                f"The file must contain exactly {len(hyperparameter_widgets)} key-value pairs."
            )
            return
        for line, (label, widget) in zip(lines, hyperparameter_widgets):
            try:
                # Extract the value after the colon, strip whitespace
                value = line.split(":", 1)[1].strip()

                # Check the type of the widget's variable and set the value appropriately
                if isinstance(widget.var, tk.DoubleVar):
                    widget.set(float(value))
                elif isinstance(widget.var, tk.IntVar):
                    widget.set(int(value))
                elif isinstance(widget.var, tk.StringVar):
                    widget.set(value)  # No conversion needed
                else:
                    raise TypeError(f"Unsupported variable type for {label}.")
            except (IndexError, ValueError, TypeError) as e:
                messagebox.showerror("File Error", f"Invalid format in line: {line.strip()} ({e})")
                return

        messagebox.showinfo("Success", "Data loaded successfully from the file!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read file: {str(e)}")

class Hyperparam:
    def __init__(self, label, value):
        self.label = label
        self.value = value

    def __repr__(self):
        return f"Hyperparam(label={self.label}, value={self.value})"

def save_parameters_to_file(hyperparameter_widgets, model_id_var):
    """
    Save the current hyperparameter values to a file in 'key: value' format.
    """
    
    # Generate a filename using Model ID and timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"{model_id_var.get()}_{timestamp}.txt"

    file_path = filedialog.asksaveasfilename(
        initialdir=DEFAULT_DIR,
        initialfile=default_filename,  # Use generated filename
        title="Save Parameters",
        defaultextension=".txt",
        filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
    )

    if not file_path:
        return  # User cancelled the dialog

    try:
        with open(file_path, "w") as file:
            for label, widget in hyperparameter_widgets:
                value = widget.get()
                file.write(f"{label.lower().replace(' ', '_')}: {value}\n")
        messagebox.showinfo("Success", f"Parameters saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save file: {str(e)}")

def save_to_last_file(hyperparameter_widgets):
    """
    Save the current hyperparameter values to the last opened file.
    If no file is open, prompt the user to select a file to save.
    """
    global last_file_path

    if not last_file_path:
        # Fallback: Use the save-as functionality if no file is open
        save_parameters_to_file(hyperparameter_widgets)
        return
    try:
        with open(last_file_path, "w") as file:
            for label, widget in hyperparameter_widgets:
                value = widget.get()
                file.write(f"{label.lower().replace(' ', '_')}: {value}\n")
            messagebox.showinfo("Success", f"Parameters saved to {last_file_path} (Model ID: {model_id_var.get()})")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save file: {str(e)}")   
     
def run_second_app():
    subprocess.Popen(
        # ["python3", "src/t_run/secoundApp.py"], 
        ["python3", "/project/src/run/side_demonstrate_s2s.py"], 
        start_new_session=True
        )   
def refresh_hyperparams():
    hyperparams = create_hyperparams(hyperparameter_widgets)
    
    # Serialize the hyperparameters to a file
    with open("hyperparams.pkl", "wb") as file:
        pickle.dump(hyperparams, file)
    
    # Create a notification file to signal the second app
    with open("notification.txt", "w") as file:
        file.write("Data Updated")
    print("Hyperparameters refreshed and sent to the second app!")     
    
def print_hyperparams(params):
    for label, value in vars(params).items():
        print(f"{label}: {value}")
    
    
from types import SimpleNamespace
import re    
def create_hyperparams(hyperparameter_widgets):
    hyperparams = []
    for label, widget in hyperparameter_widgets:
        value = widget.get()
        hyperparams.append(Hyperparam(label, value))
        
    # Convert labels to valid Python attribute names
    def format_label(label):
        label = re.sub(r'\W|^(?=\d)', '_', label).lower()
        return label.rstrip('_')
    params = SimpleNamespace(**{format_label(param.label): param.value for param in hyperparams})
    print_hyperparams(params)        

    # params = SimpleNamespace(**{param.label: param.value for param in hyperparams})
    
    # Print all collected hyperparameters
    # for param in hyperparams:
    #     print(param)  # Example output: Hyperparam(label=learning_rate, value=0.01)
    
    return params  # Return the list of Hyperparam objects         

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hyperparameter Entry Fields")
    
    # Save button for directly updating the last opened file
    direct_save_button = ttk.Button(
        root, text="Save to Last Opened File",
        command=lambda: save_to_last_file(hyperparameter_widgets)
    )
    direct_save_button.grid(row=2, column=0, padx=10, pady=10)
    

    # Create hyperparameter entry fields
    hyperparameter_widgets, model_id_var  = create_hyperparameter_fields(root)


    # Button to load data from a file
    load_button = ttk.Button(
        root, text="Load Hyperparameters from File",
        command=lambda: open_file_and_load_data(hyperparameter_widgets)
    )
    load_button.grid(row=3, column=0, padx=10, pady=10)

    # Button to print all hyperparameter values
    def print_values():
        for label, widget in hyperparameter_widgets:
            print(f"{label}: {widget.get()}")

    print_button = ttk.Button(root, text="Print Values", command=print_values)
    print_button.grid(row=4, column=0, padx=10, pady=10)
    
    # Button to save data to a file
    save_button = ttk.Button(
        root, text="Save Hyperparameters to File",
        command=lambda: save_parameters_to_file(hyperparameter_widgets, model_id_var)
    )
    save_button.grid(row=5, column=0, padx=10, pady=10)
    
    # Create a button to launch the other app
    
    open_app = tk.Button(root, text="Run Second App", command=run_second_app)
    open_app.grid(row=7, column=0, padx=10, pady=10)
    refresh_app = tk.Button(root, text="Refresh Hyperparameters", command=refresh_hyperparams)
    refresh_app.grid(row=8, column=0, padx=10, pady=10)


    root.mainloop()
