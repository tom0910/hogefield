import torchaudio.transforms as T
from core.CustomMelScale import CustomMelScale
from core.AudioSample import AudioSample

class MelSpectrogramConfig:
    """
    Stores and manages the configuration settings for the Mel spectrogram.
    """

    def __init__(self, audio_sample, n_fft, hop_length, n_mels, f_min, f_max, filter_type, toggle_mel_filter,  power=1):
        self.audio_sample = audio_sample  # Store the provided AudioSample object
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.filter_type = filter_type
        self.toggle_mel_filter = toggle_mel_filter

    # Audio Sample Rate property
    @property
    def sample_rate(self):
        """Dynamically fetch the sample rate from the current audio sample whenever accessed."""
        _, rate = self.audio_sample.load_waveform()
        return rate
        
    # step_duration property
    @property
    def step_duration(self):
        """Calculate step_duration dynamically based on the latest audio sample."""
        if hasattr(self.audio_sample, "n_sample") and self.sample_rate:
            return self.audio_sample.n_sample / self.hop_length
        return None        

    @property
    def time_resolution(self):
        """Calculates time resolution in sec based on the current sample rate and hop length."""
        return (1 / self.sample_rate) * self.hop_length
    
    def create_standard_transform(self):
        """
        Create and return a standard MelSpectrogram transform based on the current configuration.
        Returns:
            T.MelSpectrogram: A Mel spectrogram transform from torchaudio.transforms.
        """
        return T.MelSpectrogram(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            power=self.power
        )

    def create_custom_transform(self):
        """
        Create and return a custom MelScale transform using CustomMelScale.
        Returns:
            CustomMelScale: A custom Mel scale transform.
        """
        return CustomMelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            n_stft=self.n_fft // 2 + 1,  # Equivalent to the number of frequency bins
            f_min=self.f_min,
            f_max=self.f_max,
            filter_type=self.filter_type
        )

    def debug_transform(self, waveform):
        """
        Debug and log details of the transform.
        Args:
            waveform (torch.Tensor): The waveform to process.
        """
        transform = self.get_transform()
        
        # Debugging the filter bank for custom transforms
        if self.filter_type == "custom" and isinstance(transform, CustomMelScale):
            print(f"Custom Filter Bank Shape: {transform.fb.shape}")
            print(f"Filter Center Frequencies: {transform.f_pts}")
            print(f"Frequency Spread: {transform.spread}")

        # Debugging the Mel spectrogram output
        mel_spectrogram = transform(waveform)
        print(f"Mel Spectrogram Shape: {mel_spectrogram.shape}")
        return mel_spectrogram

    def get_transform(self):
        """
        Selects and returns the appropriate Mel spectrogram transform based on the current filter_type.
        Returns:
            Transform: Either a standard or custom Mel spectrogram transform.
        """
        if self.filter_type == "custom":
            return self.create_custom_transform()
        else:
            return self.create_standard_transform()

    def update_from_widgets(self, n_fft, hop_length, n_mels, f_min, f_max, power, filter_type, toggle_mel_filter):
        """
        Update the configuration based on widget values.
        Args:
            n_fft (int): The new FFT size.
            hop_length (int): The new hop length.
            n_mels (int): The new number of Mel bands.
            f_min (float): The new minimum frequency.
            f_max (float): The new maximum frequency.
            power (float): The power scaling for the spectrogram.
            filter_type (str): Type of filter to use ('standard' or 'custom').
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.filter_type = filter_type
        self.toggle_mel_filter = toggle_mel_filter

