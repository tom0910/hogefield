# File related settings
BASE_PATH = '../SpeechCommands/speech_commands_v0.02'
DEFAULT_DIRECTORY = 'backward'
DEFAULT_FILE_INDEX = 0

# Mel Spectrogram settings
DEFAULT_HOP_LENGTH = 20
DEFAULT_N_MELS = 22
DEFAULT_F_MIN = 0
DEFAULT_F_MAX = 8000
DEFAULT_N_FFT = 512
DEFAULT_POWER = 1.0 #this produces magnitude, .abs() would be redundant code for spectrogram

# Spectrogram and Spiking Settings
DEFAULT_THRESHOLD = 0.03 # in range [0..1] when melspectrogram is normalized
