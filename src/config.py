## Configuration file containing all the important variables
# size of the image after resizing
SIZE = 384

# batch size
BATCH_SIZE = 16

# number of processes to be used for pytorch dataloaders
NUM_WORKERS = 4

# optimizer variables
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-3
SCHEDULER = "CosineAnnealingLR"

# number of training rounds
EPOCHS = 6

# name of the model
MODEL_NAME = "rexnet_130"
# MODEL_NAME = "test_cnn_model"

# number of cross-validation folds
FOLDS = 4

# important directory paths
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"

# mixup augmentations
use_mixup = True
mixup_alpha = 1.0
