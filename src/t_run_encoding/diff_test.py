import torch

# Example tensor `cusm`
cusm = torch.tensor([[1.0, 2.0, 4.0],
                     [3.0, 6.0, 9.0]])

print(cusm)

# Compute differences along the last dimension
dim = -1

# `torch.diff` computes differences along the last dimension
diff = torch.diff(cusm, dim=dim)
print("torch.diff result:")
print(diff)

# Prepending zeros to match dimensions
prepend = torch.zeros_like(cusm[..., :1])  # Prepends a column of zeros
diff_with_prepend = torch.diff(cusm, dim=dim, prepend=prepend)
print("\nResult after prepending zeros:")
print(diff_with_prepend)

# Assume delta_t = 2 for demonstration
delta_t = 2.0
result = diff_with_prepend / delta_t
print("\nFinal result (discrete derivative):")
print(result)
