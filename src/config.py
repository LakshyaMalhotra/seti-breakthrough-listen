## Configuration file containing all the important variables
# size of the image after resizing
SIZE = 256

# batch size
BATCH_SIZE = 32

# number of processes to be used for pytorch dataloaders
NUM_WORKERS = 4

# optimizer variables
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-6

# number of training rounds
EPOCHS = 5

# name of the model
MODEL_NAME = "efficientnetv2_rw_s"

# number of cross-validation folds
FOLDS = 4

# important directory paths
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
