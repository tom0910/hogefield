import pickle
import os
import time
import utils.data_transfer_util as dt_util

def custom_observer():
    dt_util.observe_notifications_from_app(callback=run_function_on_refresh)

def run_function_on_refresh(hyperparams):
    print("Function started with refreshed data:")
    for label, value in vars(hyperparams).items():
        print(f"{label}: {value}")

if __name__ == "__main__":
    custom_observer()
