# Define all global params here

# Number of epochs
NUM_EPOCHS = 40
# Batch size
N_BATCH = 50
# Max sequence length
#MAX_LENGTH = 145
MAX_LENGTH = 400
# Dimensionality of character lookup
CHAR_DIM = 250
# Initialization scale
SCALE = 0.1
# Dimensionality of C2W hidden states
C2W_HDIM = 500
# Dimensionality of word vectors
WDIM = 400
# Number of classes
MAX_CLASSES = 60
# Learning rate, nesterov_momentum: 0.01, RMSProp:0.001
LEARNING_RATE = 0.001
# Display frequency
DISPF = 20
# Save frequency
SAVEF = 2400
# Regularization
REGULARIZATION = 0.0001
# Reload
RELOAD_MODEL = False
# NAG
MOMENTUM = 0.9
# clipping
GRAD_CLIP = 30.
# use bias
BIAS = False
# use schedule
SCHEDULE = True
# use rmsprop
RMSPROP = True
