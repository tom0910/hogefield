from core.AudioSample import AudioSample
import config.config as C
from core.MelSpectrogramConfig import MelSpectrogramConfig
import torch
import torchaudio.transforms as T
import os
from snntorch import functional as SNNF 
import utils.loading_functional as FL
from torch.utils.data import DataLoader
import glob  

def calculate_num_frames(L, n_fft, hop_length, center=True, show=False):
    """
    Calculate the number of time frames in the spectrogram.

    Args:
    - L: Length of the .wav input signal (samples).
    - n_fft: Window size.
    - hop_length: Hop length.
    - center: Whether padding is applied to the input signal (default: True).

    Returns:
    - Number of time frames.
    """
    if center:
        # Calculate the padded length
        padded_length = L + 2 * (n_fft // 2)
        num_frames = ((padded_length - n_fft) // hop_length) + 1
    else:
        num_frames = ((L - n_fft) // hop_length) + 1
    if show:
        print("a spektrogram hossza (= lehet a timestep):",num_frames)
        
    print(f"L={L} (type: {type(L)}), n_fft={n_fft} (type: {type(n_fft)}), hop_length={hop_length} (type: {type(hop_length)})")
    print(f"num_frames={num_frames} (type: {type(num_frames)})")
    
    return num_frames

def calculate_num_of_frames_constant():
    audio_sample = AudioSample(C.BASE_PATH, C.DEFAULT_DIRECTORY, C.DEFAULT_FILE_INDEX)
    mel_config = MelSpectrogramConfig(
        audio_sample,
        n_fft=C.DEFAULT_N_FFT,
        hop_length=C.DEFAULT_HOP_LENGTH,
        n_mels=C.DEFAULT_N_MELS,
        f_min=C.DEFAULT_F_MIN,
        f_max=C.DEFAULT_F_MAX,
        power=C.DEFAULT_POWER,
        filter_type=C.DEFAULT_FILTER_CHOICE,
        toggle_mel_filter="spktgrm"
    )
    timestep = calculate_num_frames(
            L=audio_sample.sample_rate,
            n_fft=mel_config.n_fft,
            hop_length=mel_config.hop_length,
            center=True,
            show=True
            )
    return timestep

import utils.loading_functional as LF    
def calculate_number_of_input_neurons():
    return C.DEFAULT_HOP_LENGTH
    
import utils

from snntorch import utils as snn_utils
def forward_pass(net, data, timestep):  #nem ezt használom 
  spk_rec = []
  snn_utils.reset(net)  # resets hidden states for all LIF neurons in net
  # ha a number of time steps biztos egyenlő a spectogramban az x tengely adatainak számával:
  #for step in range(data.size(0)):  # data.size(0) = number of time steps
  for step in range(timestep):
      # print("Input Data's' shape:", data.shape) #Input Data's' shape: torch.Size([201, 64, 80]) = [time_step, batch_size, neural_number], neural_number is iput size
      # print("data.size(0) is:",data.size(0)) # data.size(0) is: 201 = time_steps
      # print("Step:", step)
      # print("Input Data[step]'s' Shape:", data[step].shape) #Input Data Shape: torch.Size([batch_size, neural_numbers]) = Input Data Shape: torch.Size([64, 80])
      # print_structure("net(data[step]):",net(data[step]))
      # print("net(data[step]):",net(data[step]).shape)
      # break
      spk_out, mem_out = net(data[step])
      spk_rec.append(spk_out)
      #print("\rspk_out.shape",spk_out.shape) #spk_out.shape torch.Size([64, 35]) = [batc_size x output_size] 
  #print("\rspk_rec.shape",torch.stack(spk_rec).shape) #spk_rec.shape torch.Size([128, 64, 35]) = [number of time steps of the SNN x batch_size x output_size]
  #plot_pipline(torch.stack(spk_rec))
  return torch.stack(spk_rec)

import torch
def batch_accuracy(loader, net, timestep, device):
    # Be careful! This evaluates accuracy on the whole dataset
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
        
        loader = iter(loader)
        for batch in loader:
            # Assuming your DataLoader returns more than three items, adjust the unpacking
            data, targets, *_ = batch  # Unpack the first two items and ignore the rest
            
            # Ensure the data is reshaped and moved to the correct device
            data = data.squeeze().permute(2, 0, 1)
            data = data.to(device)
            targets = targets.to(device)
            
            # Perform the forward pass
            spk_rec = forward_pass(net, data, timestep)
            
            # Calculate accuracy for the current batch
            acc += SNNF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

        # Calculate and print the average accuracy
        avg_acc = acc / total
        print("\nBatch average accuracy: ", avg_acc)

    return avg_acc

def create_optimizer(net_params, learning_rate, num_classes, betas=(0.9, 0.999)): 
    # print(f"model params: {net_params}")
    optimizer = torch.optim.Adam(net_params, learning_rate, betas)
    # loss_fn = SNNF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2,population_code=False, num_classes=34)
    loss_fn = SNNF.mse_count_loss(correct_rate=0.75, incorrect_rate=0.25,population_code=False)
    return optimizer, loss_fn

def prepare_dataset(pth_file_path, params):
    # Prepare datasets
    train_set, test_set, target_labels = FL.prepare_datasets(
        data_path="/project/data/GSC/",
        check_labels=False
    )
    
    batch_size = params["batch_size"]
    
    # Create DataLoader with custom_collate_fn
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=int(batch_size),  # Use batch size from parameters
        shuffle=True,
        collate_fn=lambda batch: FL.custom_collate_fn(
            batch=batch,
            params=params,
            target_labels=target_labels,
            pth_file_path=pth_file_path
        )
    )
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=int(batch_size),  # Use batch size from parameters
        shuffle=False,
        collate_fn=lambda batch: FL.custom_collate_fn(
            batch=batch,
            params=params,
            target_labels=target_labels,
            pth_file_path=pth_file_path
        )
    )
    return train_loader, test_loader, 

# def save_checkpoint(checkpoint, checkpoint_dir, filename):
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#         print(f"Directory created: {checkpoint_dir}")
#     filepath = os.path.join(checkpoint_dir, filename)
#     torch.save(checkpoint, filepath)
#     # print(f"Checkpoint saved: {filepath}")
    

def save_checkpoint(checkpoint, checkpoint_dir, filename):
    """
    Save the checkpoint and keep only the most recent one.

    Args:
        checkpoint: The checkpoint object to save.
        checkpoint_dir: Directory to save the checkpoint.
        filename: The filename for the new checkpoint.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Directory created: {checkpoint_dir}")

    # Path for the new checkpoint file
    new_file_path = os.path.join(checkpoint_dir, filename)

    # Find the most recent existing checkpoint
    existing_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth")), key=os.path.getmtime)
    old_file_path = existing_files[-1] if existing_files else None

    try:
        # Save the new checkpoint
        torch.save(checkpoint, new_file_path)
        # print(f"Checkpoint saved: {new_file_path}")
        
        # Safely delete the old checkpoint if a new one was successfully created
        if old_file_path and old_file_path != new_file_path:
            os.remove(old_file_path)
            print(f"Removed old checkpoint: {old_file_path}")
    except Exception as e:
        # Handle any errors in saving the checkpoint
        print(f"Error saving checkpoint: {e}")
    

def load_latest_checkpoint(checkpoint_dir, model, optimizer):
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pth')), key=os.path.getmtime)
    if not checkpoint_files:
        return None, 0, [], []

    latest_checkpoint = checkpoint_files[-1]
    # print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint)
    model.net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint, checkpoint["epoch"], checkpoint["loss_hist"], checkpoint["acc_hist"]

# def plot_training(fig, ax1, ax2, loss_hist, acc_hist, plots_dir, filename):
#     ax1.clear()
#     ax2.clear()
#     ax1.plot(loss_hist, color="blue")
#     ax2.plot(acc_hist, color="red")
#     ax1.set_title("Training Loss")
#     ax2.set_title("Training Accuracy")
#     # Ensure the directory exists
#     os.makedirs(plots_dir, exist_ok=True)
#     # save
#     fig.savefig(os.path.join(plots_dir, filename))
#     print(f"Plot saved: {os.path.join(plots_dir, filename)}")
    
# also deletes previous
import os
import glob

def plot_training(fig, ax1, ax2, loss_hist, acc_hist, plots_dir, filename):
    """
    Plot training metrics and save the most recent plot safely.

    Args:
        fig: The matplotlib figure object.
        ax1: Axis for loss plot.
        ax2: Axis for accuracy plot.
        loss_hist: List of loss values.
        acc_hist: List of accuracy values.
        plots_dir: Directory to save plots.
        filename: Filename for the new plot.
    """
    ax1.clear()
    ax2.clear()
    ax1.plot(loss_hist, color="blue")
    ax2.plot(acc_hist, color="red")
    ax1.set_title("Training Loss")
    ax2.set_title("Training Accuracy")
    
    # Ensure the directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    # Path for the new file
    new_file_path = os.path.join(plots_dir, filename)

    # Look for an old file
    existing_files = sorted(glob.glob(os.path.join(plots_dir, "*.png")), key=os.path.getmtime)
    old_file_path = existing_files[-1] if existing_files else None

    try:
        # Save the new plot
        fig.savefig(new_file_path)
        print(f"Plot saved: {new_file_path}")
        
        # Safely delete the old file if a new one was successfully created
        if old_file_path and old_file_path != new_file_path:
            os.remove(old_file_path)
            print(f"Removed old plot: {old_file_path}")
    except Exception as e:
        # Handle any errors in saving the plot
        print(f"Error saving plot: {e}")




