import torch

def add_pth_values(pth_file_path, output_file_path, new_values):
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
        for key, value in new_values.items():
            checkpoint[key] = value
            print(f"Set '{key}' to {value}.")

        # Save the updated checkpoint
        torch.save(checkpoint, output_file_path)
        print(f"Updated checkpoint saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found - {pth_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage
pth_file_path = "/project/hyperparam/trial_rate_chng/epoch_@48prcnt.pth"  # Input file path
output_file_path = "/project/hyperparam/trial_rate_chng/epoch_@48prcnt_rated.pth"  # Output file path
new_values = {
    "correct_rate": 1,  # Add correct_rate
    "incorrect_rate": 0,  # Add incorrect_rate
}

add_pth_values(pth_file_path, output_file_path, new_values)

