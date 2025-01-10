import torch

# Original normalization function
def normalize(obj, normalize_to=None):
    """Normalize the spectrogram or cumsum globally to a 0-1 range."""
    normalized = (obj - obj.min()) / (obj.max() - obj.min())
    if normalize_to:
        normalized = normalized * normalize_to
    return normalized

# Inverse normalization function
def inverse_normalize(normalized_obj, original_min, original_max, normalize_to=None):
    """Invert the normalization to restore the original range."""
    if normalize_to:
        normalized_obj = normalized_obj / normalize_to
    return normalized_obj * (original_max - original_min) + original_min

# Test data
original_obj = torch.tensor([10.0, 20.0, 30.0, 40.0])
normalize_to = 1  # Example scaling factor

# Step 1: Normalize
normalized_obj = normalize(original_obj, normalize_to=normalize_to)

# Step 2: Invert normalization
restored_obj = inverse_normalize(
    normalized_obj,
    original_min=original_obj.min(),
    original_max=original_obj.max(),
    normalize_to=normalize_to
)

# Step 3: Print results for validation
print("Original Object:", original_obj)
print("Normalized Object:", normalized_obj)
print("Restored Object:", restored_obj)

# Check if the restored object matches the original
is_close = torch.allclose(original_obj, restored_obj, atol=1e-6)
print(f"Restoration successful: {is_close}")

