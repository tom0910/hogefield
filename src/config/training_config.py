import config.config as C

# Model hyperparameters
DEFAULT_NUM_INPUTS = C.DEFAULT_N_MELS
DEFAULT_NUM_HIDDEN = 256
DEFAULT_NUM_OUTPUTS = 35
DEFAULT_BETA_LIF = 0.6  # Decay parameter for LIF neurons
DEFAULT_THRESHOLD_LIF = 0.3  # Firing threshold for LIF neurons
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_EPOCHS = 10