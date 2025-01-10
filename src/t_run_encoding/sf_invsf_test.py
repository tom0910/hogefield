import torch
import matplotlib.pyplot as plt
import math

def normalize(obj, normalize_to=None):
    """Normalize the spectrogram or cumsum globally to a 0-1 range."""
    normalize = (obj - obj.min()) / (obj.max() - obj.min())
    if normalize_to:
        normalize=normalize*normalize_to         
    return normalize, obj.min(), obj.max()

def inverse_normalize(normalized_obj, original_min, original_max, normalize_to=None):
    """Invert the normalization to restore the original range."""
    if normalize_to:
        normalized_obj = normalized_obj / normalize_to
    return normalized_obj * (original_max - original_min) + original_min

def compute_cumulative_sum(signal, delta_t):
    """
    Compute the cumulative sum of a signal and scale by delta_t.
    Args:
        signal (torch.Tensor): The input signal.
        delta_t (float): Scaling factor.
    Returns:
        torch.Tensor: Cumulative sum of the signal.
    """
    return torch.cumsum(signal, dim=-1) * delta_t

from scipy.ndimage import gaussian_filter1d
def smooth_cumulative_sum(csum, sigma):
    """
    Smooth a cumulative sum using Gaussian smoothing.

    Args:
        csum (torch.Tensor): The input cumulative sum tensor.
        sigma (float): The standard deviation for the Gaussian kernel.

    Returns:
        torch.Tensor: The smoothed cumulative sum.
    """
    # Ensure input is a 1D or 2D tensor
    if csum.ndim == 1:
        csum = csum.unsqueeze(0)  # Add a batch dimension if necessary

    # Apply Gaussian smoothing along the last dimension
    smoothed = torch.tensor(gaussian_filter1d(csum.numpy(), sigma=sigma, axis=-1))

    return smoothed

def differentiate_smoothed_csum(smoothed_csum, delta_t):
    """
    Differentiate the smoothed cumulative sum.

    Args:
        smoothed_csum (torch.Tensor): The smoothed cumulative sum.
        delta_t (float): The time step between samples.

    Returns:
        torch.Tensor: The derivative of the smoothed cumulative sum.
    """
    # Calculate the differences along the last dimension
    delta_y = torch.diff(smoothed_csum, dim=-1, prepend=torch.zeros_like(smoothed_csum[..., :1]))
    
    # Divide by delta_t to compute the derivative
    derivative = delta_y / delta_t
    
    return derivative


def reconstruct_from_cumulative_sum(cusm, delta_t):
    """
    Reconstruct the original signal from its cumulative sum.
    Args:
        cusm (torch.Tensor): The cumulative sum of the signal.
        delta_t (float): Scaling factor used during cumulative sum computation.
    Returns:
        torch.Tensor: Reconstructed original signal.
    """
    return torch.diff(cusm, dim=-1, prepend=torch.zeros_like(cusm[..., :1])) / delta_t

# Define step-forward encoding and its inverse
def step_forward_encoding(batch, thr, neg=False):
    L = batch.shape[-1]
    base = batch[..., 0]
    out = torch.zeros_like(batch)
    base_sig = torch.zeros_like(batch)
    # Main encoding loop
    for t in range(1, L):
        # print(f"is string?{type(thr)} and {type(base)}")
        spikes = torch.where(batch[..., t] > base + thr, 1, 0) - torch.where(batch[..., t] < base - thr, 1, 0)
        base += spikes * thr
        out[..., t] = spikes
        base_sig[..., t] = base
    # Separate positive and negative spike signals
    pos_sig = torch.clamp(out, min=0)
    neg_sig = torch.where(out < 0, out if neg else -out, torch.zeros_like(out))
    
    return base_sig, pos_sig, neg_sig

def inverse_step_forward(pos_sig, thr):
    base = torch.zeros_like(pos_sig[..., 0])
    csum = torch.zeros_like(pos_sig)
    for t in range(1, pos_sig.shape[-1]):
        base += pos_sig[..., t] * thr
        csum[..., t] = base
    return csum


# Generate a sine wave as input
time_steps = 1000
amplitude = 1
frequency = 5
sampling_rate = 1000 # Hz

t = torch.linspace(0, 1, time_steps)
sine_wave = amplitude + amplitude * torch.sin(2 * math.pi * frequency * t)   
print("sine_wave.shape:",sine_wave.shape)

# 1-NORMALIZE
sine_wave_norm, signal_min, signal_max = normalize(sine_wave)

sine_wave_for_1_neuron = sine_wave_norm.unsqueeze(0)
#calculate delta_t for integration
delta_t = 1 / sampling_rate
# 2-CUMULATIVE SUM:
cusm=compute_cumulative_sum(signal=sine_wave_for_1_neuron,delta_t=delta_t)
# cusm = torch.cumsum(sine_wave_for_1_neuron, dim=-1) * delta_t

# Test step-forward encoding and its inverse
thr = 0.01  # Set threshold
print("input shape to sf:",cusm.shape)
# 3-SPIKE GENERATION
_,spikes,_ = step_forward_encoding(cusm, thr)
print(f"spikes type from generate_spikes() func:{type(spikes)} and shape:{spikes.shape}")
# 4-INVERSE SPIKE GENERATION (RECONSTRUC CUMULATIVE SUM)
reconstructed_csum = inverse_step_forward(pos_sig=spikes, thr=thr)

# check closeness of reconstruction:
# is_close = torch.allclose(cusm, reconstructed_csum, atol=5e-1)
# print(f"Are cusm and reconstructed_csum close? {is_close}")

# 5-INVERSE CUMULATIVE SUM
smoothed_cusm=smooth_cumulative_sum(csum=reconstructed_csum, sigma=20)
normalized_reconstructed_sine_wave=differentiate_smoothed_csum(smoothed_csum=smoothed_cusm, delta_t=delta_t)
# reconstruct_from_cumulative_sum(cusm=reconstructed_csum, delta_t=delta_t)
#  INVERSE NORMALIZATION
reconstructed_sine_wave=inverse_normalize(normalized_reconstructed_sine_wave,signal_min,signal_max)


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(t, sine_wave, label="1. Original Sine Wave", color="blue", linestyle="--")
plt.plot(t, sine_wave_norm, label="2. Normalized Sine Wave", color="yellow", linestyle="-")
plt.plot(t, cusm[-1], label="3. Csum Wave", color="black", linestyle="--")
# print("csum shape:",cusm.shape)
plt.scatter(t, spikes.squeeze(), label="4. Spikes", color="red", marker="o", s=10)
plt.plot(t, reconstructed_csum[-1], label="5. Reconstructed Csum", color="blue", alpha=0.7)
plt.plot(t, smoothed_cusm[-1], label="6. Smoothed Csum", color="red", alpha=1)

plt.plot(t, normalized_reconstructed_sine_wave[-1], label="7. Reconstructed Normalized Sine Wave ", color="black", alpha=0.5, linestyle='-')
plt.plot(t, reconstructed_sine_wave[-1], label="8. Reconstructed Sine Wave ", color="red", alpha=0.5, linestyle='-')
# print("shape of normzd sine wave:",normalized_reconstructed_sine_wave.squeeze().shape)
# print("shape of normzd sine wave:",normalized_reconstructed_sine_wave[-1].shape)


plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Step-Forward Encoding and Reconstruction")
plt.grid(True)
plt.ylim(0, 3)
plt.show()


