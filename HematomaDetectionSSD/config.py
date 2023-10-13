import torch

BATCH_SIZE = 16 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 20 # Number of epochs to train for.   # AVISHA: this used to be 75
NUM_WORKERS = 4 # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/train'
# Validation images and XML files directory.
VALID_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/val'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'hematoma'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = True

# Location to save model and plots.
OUT_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/HematomaDetectionSSD/outputs'