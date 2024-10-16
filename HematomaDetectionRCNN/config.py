import torch

import nni

hyp_params = {'NUM_EPOCHS': 60, 
                  'lr': 0.004934620401006168, 
                  'BATCH_SIZE': 4}


#0.498

################ NNI ###########################

# optimized_params = nni.get_next_parameter()
# hyp_params.update(optimized_params)

##################### NNI ########################

BATCH_SIZE = hyp_params['BATCH_SIZE'] # Increase / decrease according to GPU memeory.
RESIZE_TO = 256 #256 for final #640 
NUM_EPOCHS = hyp_params['NUM_EPOCHS'] # Number of epochs to train for.   # AVISHA: this used to be 75
NUM_WORKERS = 0 # Number of parallel workers for data loading.
LR = hyp_params['lr']



# BATCH_SIZE = 8 # increase / decrease according to GPU memeory
# RESIZE_TO = 416 # resize the image for training and transforms
# NUM_EPOCHS = 20 # number of epochs to train for
# NUM_WORKERS = 0

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#DEVICE = torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/train'
# validation images and XML files directory
VALID_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/val'

TEST_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/test'
# classes: 0 index is reserved for background
CLASSES = [
    '__background__', 'hematoma'
]
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/HematomaDetectionRCNN/outputs'