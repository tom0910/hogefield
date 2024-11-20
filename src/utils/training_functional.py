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
    return num_frames
    