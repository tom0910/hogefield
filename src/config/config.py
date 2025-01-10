# File related settings
BASE_PATH = 'data/GSC/SpeechCommands/speech_commands_v0.02/'
DEFAULT_DIRECTORY = 'backward'
DEFAULT_FILE_INDEX = 0
DEFAULT_MATLAB_DIRECTORY="/project/results2"

# Mel Spectrogram settings
DEFAULT_HOP_LENGTH = 20
DEFAULT_N_MELS = 22
DEFAULT_F_MIN = 0
DEFAULT_F_MAX = 8000
DEFAULT_N_FFT = 512
DEFAULT_POWER = 1.0 #this produces magnitude, .abs() would be redundant code for spectrogram

# Spectrogram and Spiking Settings
DEFAULT_THRESHOLD_MIN  =0.000001
DEFAULT_THRESHOLD_MAX  =0.001
DEFAULT_THRESHOLD_STEP =0.000001
DEFAULT_THRESHOLD      = 0.0000420 # in range [0..1] when melspectrogram is normalized

DEFAULT_FILTER_CHOICE="standard"
FILTER_VALUE1 = "standard"
FILTER_VALUE2 = "custom"
FILTER_VALUE3  = "narrowband"

DEFAULT_FILTER_SPCTRGRM_PLT_CHOICE="sptrgm"
DEFAULT_SPCTRGRM_PLT="sptrgm"
DEFAULT_FILTER_PLT="filter"

DEFAULT_SPIKE_PLT_PICK="spikes"
DEFAULT_DIST_PLT_PICK = "distribution"

#network
BATCH_SIZE=256
NEARUN_PER_LAYER=256
LAYER_NUMBER=4
NUMBER_OF_INPUTS_TO_NN = 20 #should be n_mel
NUMBER_OF_OUTPUTS_OF_NN = 35
NUM_HIDDEN_NEURONS = 256
BETA_LIF = 0.9
THRESOLD_LIF = 0.5
DEVICE="cuda"
LEARNING_RATE = 2e-4
