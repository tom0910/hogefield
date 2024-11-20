import tkinter as tk
from tkinter import ttk

class EntryWidget:
    def __init__(self, parent, variable, label_text="Enter:", default_value=10):
        self.var = variable
        # Dynamically set the default value
        if isinstance(self.var, tk.StringVar):
            self.var.set(str(default_value))
        elif isinstance(self.var, (tk.IntVar, tk.DoubleVar)):
            self.var.set(default_value)
        else:
            raise TypeError("Unsupported variable type: Use StringVar, IntVar, or DoubleVar.")
        
        print(f"Default value set: {self.var.get()}")
        self.label = ttk.Label(parent, text=label_text)
        self.entry = ttk.Entry(parent, textvariable=self.var)

    def grid(self, row, column, **kwargs):
        self.label.grid(row=row, column=column, **kwargs)
        self.entry.grid(row=row, column=column + 1, **kwargs)

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(value)

    def bind(self, callback):
        if callback is not None:
            self.var.trace_add("write", lambda *args: callback(self.get()))
        else:
            raise ValueError("Callback function cannot be None")

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("EntryWidget Example")

    int_var = tk.IntVar()
    widget = EntryWidget(root, int_var, "Enter a number:", default_value=42)
    widget.grid(0, 0)

    # Button to fetch the value
    def show_value():
        print(f"Value in Entry: {widget.get()}")

    btn = ttk.Button(root, text="Show Value", command=show_value)
    btn.grid(row=1, column=0) 

    root.mainloop()
