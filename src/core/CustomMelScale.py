import torch
# from torchaudio.functional.functional import hz_to_mel, mel_to_hz
from torch import Tensor
from torchaudio.transforms import MelScale
from typing import List, Optional, Tuple, Union
import math

class CustomMelScale(MelScale):
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        filter_type: str = "standard",
        spread: int = 100,  # Example spread
    ) -> None:
        super().__init__(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_stft,
            norm=norm,
            mel_scale=mel_scale,
        )
        
        # Initialize attributes
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.norm = norm
        self.mel_scale = mel_scale    

        # Set attributes before calling update_filter_bank
        self.n_stft = n_stft
        self.filter_type = filter_type
        self.spread = spread
        
        # Calculate all_freqs and f_pts and store as attributes
        self.all_freqs = torch.linspace(0, sample_rate // 2, n_stft)
        self.f_pts = torch.linspace(f_min, f_max, n_mels).tolist()  # Example center frequencies

        # Set the filter type and initialize the filter bank accordingly
        self.update_filter_bank()

    def update_filter_bank(self):
        """Update filter bank based on filter_type."""
        if self.filter_type == "custom":
            fb = custom_melscale_fbanks(
                n_freqs=self.n_stft,
                f_min=self.f_min,
                f_max=self.f_max,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
                norm=self.norm,
                mel_scale=self.mel_scale,
            )
            
        elif self.filter_type == "narrowband":
            fb = standard_narrowband(
                n_freqs=self.n_stft,
                f_min=self.f_min,
                f_max=self.f_max,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
                norm=self.norm,
                mel_scale=self.mel_scale,
                spread=self.spread
            )
        else:
            fb = self.fb  # Use the original filter bank from MelScale

        self.register_buffer("fb", fb)

    def set_filter_type(self, new_filter_type: str):
        """Change filter type and update the filter bank."""
        if new_filter_type not in ["standard", "custom", "narrowband"]:
            raise ValueError("filter_type must be 'standard' or 'custom' or 'narrowband'")
        self.filter_type = new_filter_type
        self.update_filter_bank()

def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    r"""Convert Hz to Mels.

    Args:
        freqs (float): Frequencies in Hz
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        mels (float): Frequency in Mels
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    print("mels:",mels)
    return mels


def _mel_to_hz(mels: Tensor, mel_scale: str = "htk") -> Tensor:
    """Convert mel bin numbers to frequencies.

    Args:
        mels (Tensor): Mel frequencies
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        freqs (Tensor): Mels converted in Hz
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs

def standard_narrowband(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    spread: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
):
    
    # Calculate all_freqs and f_pts and store as attributes
    # self.all_freqs = torch.linspace(0, sample_rate // 2, n_stft)
    # self.f_pts = torch.linspace(f_min, f_max, n_mels).tolist()  # Example center frequencies
    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = torch.linspace(m_min, m_max, n_mels) # +2
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)
    spread = 2*(sample_rate//2)/(n_freqs-1) # Spread of 3 Hz on either side of each midpoint
    fb = create_triangular_filterbank_variable_spread(all_freqs=all_freqs,f_pts=f_pts,spread=spread)
    
    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    return fb

def custom_melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> torch.Tensor:
    # print(f"in custum filterbank values are: {n_freqs},{f_min},{f_max},{n_mels},{sample_rate},{norm},{mel_scale}")
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    
    # f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)
    f_pts = [275, 375, 490, 615, 735, 857, 980, 1110, 1225, 1405, 1660, 1895, 2145, 3735, 6365, 7625]
    
    # Spread in Hz (the triangle will span midpoint +/- spread)
    # very minimu spread is calculated here:
    spread = 2*(sample_rate//2)/(n_freqs-1) # Spread of 3 Hz on either side of each midpoint
    
    # print(f"all_freqs:{all_freqs[0:3]}, few f_pts: {f_pts[0:3]}, spread: {spread} , sample_rate//2: {sample_rate//2}, n_freqs: {n_freqs}")
    
    # Use your custom triangular filterbank
    fb = create_triangular_filterbank_variable_spread(all_freqs, f_pts, spread)
    # original code:
    #fb = _create_triangular_filterbank(all_freqs, f_pts)

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    return fb

import torch

def my_custom_triangular_filterbank(all_freqs: torch.Tensor, f_pts: torch.Tensor) -> torch.Tensor:
    # Custom filterbank logic
    # Example: A simple random matrix, replace this with your actual logic
    custom_fb = torch.randn(all_freqs.size(0), f_pts.size(0) - 2)
    return custom_fb

def create_triangular_filterbank_variable_spread(all_freqs: torch.Tensor, f_pts: torch.Tensor, spread: int) -> torch.Tensor:
    """
    Create a triangular filter bank with a defined spread for each filter.
    
    Args:
        all_freqs (torch.Tensor): Frequencies corresponding to the rows in the filter bank (e.g., from STFT).
        f_pts (torch.Tensor): Center frequencies of the triangular filters.
        spread (int): Width of the base of each triangle (in Hz).

    Returns:
        torch.Tensor: A filter bank matrix of size (n_freqs, n_filters), where each column is a filter.
    """
    n_freqs = len(all_freqs)  # Number of frequency bins
    n_filters = len(f_pts)   # Number of filters

    # Initialize the filter bank as a zero matrix
    filter_bank = torch.zeros(n_freqs, n_filters)

    # Loop over each filter
    for filter_idx, center_freq in enumerate(f_pts):
        # Calculate the triangular shape for this filter
        for freq_idx, freq in enumerate(all_freqs):
            if center_freq - spread <= freq <= center_freq:
                # Rising slope of the triangle
                filter_bank[freq_idx, filter_idx] = (freq - (center_freq - spread)) / spread
            elif center_freq < freq <= center_freq + spread:
                # Falling slope of the triangle
                filter_bank[freq_idx, filter_idx] = 1 - (freq - center_freq) / spread

    return filter_bank

# def create_triangular_filterbank_variable_spread(all_freqs: torch.Tensor, f_pts: torch.Tensor, spread: int) -> torch.Tensor:
#     """
#     Create a triangular filter bank with a defined spread for each filter.
    
#     Args:
#         all_freqs (torch.Tensor): Frequency points (e.g., from STFT).
#         f_pts (torch.Tensor): Midpoints where each triangle peaks.
#         spread (int): The width of the triangle's base on either side of the midpoint (in Hz).
    
#     Returns:
#         torch.Tensor: The filter bank of size (n_freqs, n_filter).
#     """
#     # Initialize the filter bank (n_freqs x n_filter)
#     fb = torch.zeros(len(all_freqs), len(f_pts))
    
#     # Create a triangular filter for each midpoint
#     # for i, midpoint in enumerate(f_pts):
#     for i in range(0, len(f_pts)):
#     # for i in range(1, len(f_pts) - 1):  # Start from the second midpoint and end at the second-to-last
#         midpoint = f_pts[i]  # Define midpoint inside the loop
#         for j, freq in enumerate(all_freqs):
#             if midpoint - spread <= freq <= midpoint:
#                 # Upward slope: from (midpoint - spread) to midpoint
#                 fb[j, i] = (freq - (midpoint - spread)) / spread
#             elif midpoint < freq <= midpoint + spread:
#                 # Downward slope: from midpoint to (midpoint + spread)
#                 fb[j, i] = 1 - (freq - midpoint) / spread
    
#     return fb

# original code here:
def _create_triangular_filterbank(
    all_freqs: Tensor,
    f_pts: Tensor,
) -> Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb  