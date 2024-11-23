import os
import time
import pickle

class Hyperparam:
    def __init__(self, label, value):
        self.label = label
        self.value = value

    def __repr__(self):
        return f"Hyperparam(label={self.label}, value={self.value})"


def observe_notifications_from_app(callback=None):
    global hyperparams_in_main
    print("Observing for updates...")
    last_mtime = None
    while True:
        if os.path.exists("notification.txt"):
            mtime = os.path.getmtime("notification.txt")
            if last_mtime is None or mtime != last_mtime:
                last_mtime = mtime
                print("Update detected!")
                hyperparams_in_main = load_hyperparams()  # Update the main variable
                if callback:  # If a callback is provided, call it with the updated hyperparameters
                    callback(hyperparams_in_main)

def load_hyperparams():
    try:
        with open("hyperparams.pkl", "rb") as file:
            hyperparams = pickle.load(file)
            print("Received Hyperparameters:")
            # for param in hyperparams:
            #     print(param)
            # print(hyperparams)
            return hyperparams  # Return the loaded hyperparameters
    except FileNotFoundError:
        print("Hyperparameter file not found.")
        return []


# def load_hyperparams():
#     try:
#         with open("hyperparams.pkl", "rb") as file:
#             hyperparams = pickle.load(file)
#             print("Received Hyperparameters:")
#             for param in hyperparams:
#                 print(param)
#             return hyperparams
#     except FileNotFoundError:
#         print("Hyperparameter file not found.")

# def observe_notifications_from_app():
#     print("Observing for updates...")
#     last_mtime = None
#     while True:
#         if os.path.exists("notification.txt"):
#             mtime = os.path.getmtime("notification.txt")
#             if last_mtime is None or mtime != last_mtime:
#                 last_mtime = mtime
#                 print("Update detected!")
#                 load_hyperparams()
#         time.sleep(1)
