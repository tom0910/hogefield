import matplotlib.pyplot as plt
from core.CustomMelScale import create_triangular_filterbank_variable_spread

filter_bank = create_triangular_filterbank_variable_spread(all_freqs, f_pts, spread)

plt.figure(figsize=(10, 6))
for i in range(filter_bank.shape[1]):
    plt.plot(all_freqs, filter_bank[:, i], label=f"Filter {i+1} (center={f_pts[i]} Hz)")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Triangular Filter Bank with Variable Spread")
plt.legend()
plt.grid(True)
plt.show()
