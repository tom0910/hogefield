import tkinter as tk
from tkinter import ttk

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
    # Hyperparameters and their default values
    hyperparameters = [
        ("Batch Size", 128, tk.IntVar),
        ("SF Threshold", 0.0015, tk.DoubleVar),
        ("Hop Length", 20, tk.IntVar),
        ("F Min", 8000, tk.IntVar),
        ("F Max", 0, tk.IntVar),
        ("N Mels", 22, tk.IntVar),
        ("N FFT", 512, tk.IntVar),
        ("Wav File Samples", 16000, tk.IntVar),
        ("Timestep", 801, tk.IntVar),
    ]

    # Create a frame for the fields
    frame = ttk.Frame(parent)
    frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    # Create EntryWidgets for each hyperparameter
    widgets = []
    for row, (label, default, var_type) in enumerate(hyperparameters):
        var = var_type(value=default)
        widget = EntryWidget(frame, variable=var, label_text=f"{label}:", default_value=default)
        widget.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        widgets.append((label, widget))

    return widgets

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hyperparameter Entry Fields")

    # Create hyperparameter entry fields
    hyperparameter_widgets = create_hyperparameter_fields(root)

    # Button to print all hyperparameter values
    def print_values():
        for label, widget in hyperparameter_widgets:
            print(f"{label}: {widget.get()}")

    btn = ttk.Button(root, text="Print Values", command=print_values)
    btn.grid(row=1, column=0, padx=10, pady=10)

    root.mainloop()
