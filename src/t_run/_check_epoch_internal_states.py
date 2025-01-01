import torch

def add_pth_values(pth_file_path):
    """
    Add or update specific values in a .pth file.

    Args:
        pth_file_path (str): Path to the input .pth file.
        output_file_path (str): Path to save the modified .pth file.
        new_values (dict): Dictionary of keys and their new values to add or update.
    """
    try:
        # Load the .pth file
        checkpoint = torch.load(pth_file_path)

        # Add or update new values
        for key, value in checkpoint.items():
            # checkpoint[key] = value
            print(f"---------- Set: '{key}' to {value}.")

        # Save the updated checkpoint
        #torch.save(checkpoint, output_file_path)
        print(f"checked file is : {pth_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found - {pth_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage
pth_file_path = "/project/hypertrain2/nb_n40_LearnBetaThr_fromfixedvalues/pth/epoch_7.pth"  # Input file path


add_pth_values(pth_file_path)

