import tkinter as tk
import subprocess
import pickle
import os
import time

class Hyperparam:
    def __init__(self, label, value):
        self.label = label
        self.value = value

    def __repr__(self):
        return f"Hyperparam(label={self.label}, value={self.value})"

# Function to create hyperparameters
def create_hyperparams(hyperparameter_widgets):
    hyperparams = []
    for label, widget in hyperparameter_widgets:
        value = widget.get()
        hyperparams.append(Hyperparam(label, value))
    return hyperparams

# Function to run the second app
def run_second_app():
    subprocess.Popen(["python3", "/project/src/t_run/secoundApp.py"], start_new_session=True)

# Function to refresh hyperparameters and notify the second app
def refresh_hyperparams():
    hyperparams = create_hyperparams(hyperparameter_widgets)
    
    # Serialize the hyperparameters to a file
    with open("hyperparams.pkl", "wb") as file:
        pickle.dump(hyperparams, file)
    
    # Create a notification file to signal the second app
    with open("notification.txt", "w") as file:
        file.write("Data Updated")
    print("Hyperparameters refreshed and sent to the second app!")


if __name__ == "__main__":
    # Main app's GUI
    root = tk.Tk()
    root.title("Main App")

    # Example widgets
    hyperparameter_widgets = [
        ("learning_rate", tk.Entry(root)),
        ("batch_size", tk.Entry(root))
    ]

    # Layout the widgets
    for label, widget in hyperparameter_widgets:
        tk.Label(root, text=label).pack()
        widget.pack()

    # Add buttons
    tk.Button(root, text="Run Second App", command=run_second_app).pack(pady=10)
    tk.Button(root, text="Refresh Hyperparameters", command=refresh_hyperparams).pack(pady=10)

    root.mainloop()
