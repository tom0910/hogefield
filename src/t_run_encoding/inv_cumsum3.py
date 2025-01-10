import torch
import matplotlib.pyplot as plt

# Generate a linearly increasing sinusoidal signal
time_steps = 100
t = torch.linspace(0, 10, time_steps)
linear_sine_wave = torch.sin(2 * torch.pi * 0.5 * t)  # Linear + sinusoidal component

delta_t = 1/time_steps

# Compute cumulative sum
cusm = torch.cumsum(linear_sine_wave, dim=-1) * delta_t

# Reconstruct original signal from cumulative sum
reconstructed_signal = torch.diff(cusm, dim=-1, prepend=torch.zeros_like(cusm[..., :1])) / delta_t

# Check if reconstruction matches the original signal
is_close = torch.allclose(linear_sine_wave, reconstructed_signal, atol=1e-6)
print(f"Reconstruction successful: {is_close}")

# Plot original and reconstructed signals
plt.figure(figsize=(10, 5))
plt.plot(t, linear_sine_wave, label="Original Signal", color="orange")
plt.plot(t, reconstructed_signal, label="Reconstructed Signal", linestyle="-.", color="blue", alpha=0.8)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Reconstruction of Linearly Increasing Sinusoid")
plt.legend()
plt.grid()
plt.show()

