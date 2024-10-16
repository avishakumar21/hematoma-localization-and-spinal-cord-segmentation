import torch
import nni 


hyp_params = {'NUM_EPOCHS': 60, 
                  'lr': 0.00009923345976519558, 
                  'BATCH_SIZE': 8}

'''NUM_EPOCHS:60
lr:0.00009923345976519558
BATCH_SIZE:8'''


################ NNI ###########################

# optimized_params = nni.get_next_parameter()
# hyp_params.update(optimized_params)

##################### NNI ########################

BATCH_SIZE = hyp_params['BATCH_SIZE'] # Increase / decrease according to GPU memeory.
RESIZE_TO = 256 
NUM_EPOCHS = hyp_params['NUM_EPOCHS'] # Number of epochs to train for.   # AVISHA: this used to be 75
NUM_WORKERS = 0 # Number of parallel workers for data loading.
LR = hyp_params['lr']

#BATCH_SIZE = 4 # Increase / decrease according to GPU memeory.
#RESIZE_TO = 256 # Resize the image for training and transforms.
#NUM_EPOCHS = 40 # Number of epochs to train for.
#NUM_WORKERS = 0 # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#DEVICE = torch.device('cpu')

# Training images and XML files directory.

TRAIN_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/train'
# Validation images and XML files directory.
VALID_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/val'

TEST_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/test'


# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'hematoma'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs'