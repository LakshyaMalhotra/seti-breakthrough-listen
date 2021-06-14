## Configuration file containing all the important variables
# size of the image after resizing
SIZE = 256

# batch size
BATCH_SIZE = 32

# number of processes to be used for pytorch dataloaders
NUM_WORKERS = 4

# optimizer variables
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-5
SCHEDULER = "CosineAnnealingLR"

# number of training rounds
EPOCHS = 4

# name of the model
MODEL_NAME = "efficientnet_b1"
# MODEL_NAME = "test_cnn_model"

# number of cross-validation folds
FOLDS = 4

# important directory paths
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
