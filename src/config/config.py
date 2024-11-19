# File related settings
BASE_PATH = 'data/GSC/SpeechCommands/speech_commands_v0.02/'
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
DEFAULT_THRESHOLD_MAX=0.03
DEFAULT_THRESHOLD_STEP =0.00001
DEFAULT_THRESHOLD = 0.00420 # in range [0..1] when melspectrogram is normalized

DEFAULT_FILTER_CHOICE="standard"
FILTER_VALUE1 = "standard"
FILTER_VALUE2 = "custom"

DEFAULT_FILTER_SPCTRGRM_PLT_CHOICE="sptrgm"
DEFAULT_SPCTRGRM_PLT="sptrgm"
DEFAULT_FILTER_PLT="filter"

DEFAULT_SPIKE_PLT_PICK="spikes"
DEFAULT_DIST_PLT_PICK = "distribution"
