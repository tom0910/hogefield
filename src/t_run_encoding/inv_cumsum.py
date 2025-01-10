import torch

# Generate a simple signal
signal = torch.linspace(0, 10, steps=100)  # Example: linearly increasing signal
delta_t = 0.1

# Compute cumulative sum
cusm = torch.cumsum(signal, dim=-1) * delta_t

# Reconstruct original signal from cumulative sum
reconstructed_signal = torch.diff(cusm, dim=-1, prepend=torch.zeros_like(cusm[..., :1])) / delta_t

# Check if the reconstruction matches the original signal
is_close = torch.allclose(signal, reconstructed_signal, atol=1e-6)
print(f"Reconstruction successful: {is_close}")

# Print results for debugging
print("Original Signal:", signal[:10])  # First 10 values
print("Reconstructed Signal:", reconstructed_signal[:10])  # First 10 values
