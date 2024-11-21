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

from core.AudioSample import AudioSample
import config.config as C
from core.MelSpectrogramConfig import MelSpectrogramConfig

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

def forward_pass(net, data, timestep):
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net
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
            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

        # Calculate and print the average accuracy
        avg_acc = acc / total
        print("\nBatch average accuracy: ", avg_acc)

    return avg_acc

import torch
import torchaudio.transforms as T
import libs.SNNnanosenpy as snnnspy

