from core.CheckPoinManager import CheckpointManager
def modify_and_save_checkpoint(pth_file_path, output_file_path, updated_hyperparameters):
    """
    Load a .pth file using CheckpointManager, update hyperparameters, and save it back.

    Args:
        pth_file_path (str): Path to the input .pth file.
        output_file_path (str): Path to save the updated .pth file.
        updated_hyperparameters (dict): Dictionary of hyperparameters to update or add.
    """
    try:
        # Load the checkpoint manager
        checkpoint_manager = CheckpointManager.load_checkpoint_with_defaults(pth_file_path)

        # Update hyperparameters
        checkpoint_manager.set_hyperparameters(updated_hyperparameters)

        # Save the updated checkpoint
        checkpoint_manager.save(output_file_path)

        print(f"Checkpoint updated and saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found - {pth_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage
pth_file_path = "/project/hyperparam/trial/tria_hyp_change3/pth/epoch_4.pth"  # Input .pth file
output_file_path = "/project/hyperparam/trial/tria_hyp_change3/pth/epoch_4_changed.pth"  # Output .pth file
updated_hyperparameters = {
    "learning_rate": 0.0003,
    "batch_size": 123,
}

modify_and_save_checkpoint(pth_file_path, output_file_path, updated_hyperparameters)
