import torch

def modify_pth_values(pth_file_path, output_file_path, keys_to_zero):
    """
    Modify specific values in a .pth file by setting them to zero.

    Args:
        pth_file_path (str): Path to the input .pth file.
        output_file_path (str): Path to save the modified .pth file.
        keys_to_zero (list): List of keys whose values should be set to zero.
    """
    try:
        # Load the .pth file
        checkpoint = torch.load(pth_file_path)

        # Modify specified keys
        for key in keys_to_zero:
            if key in checkpoint:
                checkpoint[key] = 0  # Set value to zero
                print(f"Set '{key}' to 0.")
            else:
                print(f"Key '{key}' not found in the checkpoint.")

        # Save the updated checkpoint
        torch.save(checkpoint, output_file_path)
        print(f"Updated checkpoint saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found - {pth_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage
pth_file_path = "/project/hyperparam/trial_rate_chng/epoch_@48prcnt_rated.pth"  # Input file path
output_file_path = "/project/hyperparam/trial_rate_chng/epoch_@48prcnt_zerod.pth"  # Output file path
keys_to_zero = ["counter", "epoch"]  # Keys to set to zero

modify_pth_values(pth_file_path, output_file_path, keys_to_zero)
