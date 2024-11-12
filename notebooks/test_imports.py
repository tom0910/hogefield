import sys
import os

# Add the `src` directory to the Python path explicitly
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


import sys
print("PYTHONPATH:", sys.path)

# Import statements based on the structure we have in `src`
from core.AudioSample import AudioSample
from src.core.MelSpectrogramConfig import MelSpectrogramConfig
import src.config.config as C
from src.config.config import BASE_PATH, DEFAULT_DIRECTORY, DEFAULT_FILE_INDEX, DEFAULT_THRESHOLD
from src.display.widgets_setup import create_widgets
from src.display.display_functions import (
    display_audio_in_widget, plot_audio_waveform_in_widget,
    plot_mel_spectrogram_in_widget, plot_spikes_in_widget, polt_reverse
)
from src.utils.widget_sync_utils import set_audio_sample_from_widget_values, set_mel_config_from_widget_values
from src.core.Spikes import Spikes

# Simple print statement to confirm successful imports
print("Imports successful!")
