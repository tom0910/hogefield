import torch
import matplotlib.pyplot as plt
from torchaudio.transforms import Spectrogram

# Generate a simple sine wave
sample_rate = 16000
freq = 440  # Frequency in Hz
duration = 1.0  # Duration in seconds
t = torch.linspace(0, duration, int(sample_rate * duration))
waveform = torch.sin(2 * torch.pi * freq * t).unsqueeze(0)  # Shape [1, time]

# Parameters for spectrogram
n_fft = 512
hop_length = 256
win_length = 512
window = torch.hann_window(win_length)

# Test different normalization methods
spectrogram_no_norm = Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=1.0, normalized=False)
spectrogram_window_norm = Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=1.0, normalized="window")
spectrogram_frame_norm = Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=1.0, normalized="frame_length")

# Compute spectrograms
spec_no_norm = spectrogram_no_norm(waveform)
spec_window_norm = spectrogram_window_norm(waveform)
spec_frame_norm = spectrogram_frame_norm(waveform)

# Plot the spectrograms
plt.figure(figsize=(18, 5))

# No normalization
plt.subplot(1, 3, 1)
plt.title("No Normalization")
plt.imshow(spec_no_norm.log2()[0].numpy(), aspect="auto", origin="lower")
plt.colorbar(label="Log Magnitude")

# Window normalization
plt.subplot(1, 3, 2)
plt.title("Window Normalization")
plt.imshow(spec_window_norm.log2()[0].numpy(), aspect="auto", origin="lower")
plt.colorbar(label="Log Magnitude")

# Frame length normalization
plt.subplot(1, 3, 3)
plt.title("Frame Length Normalization")
plt.imshow(spec_frame_norm.log2()[0].numpy(), aspect="auto", origin="lower")
plt.colorbar(label="Log Magnitude")

plt.tight_layout()
plt.show()
