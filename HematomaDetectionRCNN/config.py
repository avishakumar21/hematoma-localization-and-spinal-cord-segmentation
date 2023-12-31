import torch
BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 416 # resize the image for training and transforms
NUM_EPOCHS = 20 # number of epochs to train for
NUM_WORKERS = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/train'
# validation images and XML files directory
VALID_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/val'

TEST_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/test'
# classes: 0 index is reserved for background
CLASSES = [
    '__background__', 'hematoma'
]
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True
# location to save model and plots
OUT_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/hematoma-localization-and-spinal-cord-segmentation/HematomaDetectionRCNN/outputs'