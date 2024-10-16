import torch
import nni

hyp_params = {'batch': 8,
               'lr': 0.00025088981591017467, 
               'epochs': 20
                  }

################ NNI ###########################

# optimized_params = nni.get_next_parameter()
# hyp_params.update(optimized_params)

##################### NNI ########################

BATCH_SIZE = hyp_params['batch'] # Increase / decrease according to GPU memeory.
#RESIZE_TO = 640 # Resize the image for training and transforms.
RESIZE_TO = 256 
NUM_EPOCHS = hyp_params['epochs'] # Number of epochs to train for.   # AVISHA: this used to be 75
NUM_WORKERS = 0 # Number of parallel workers for data loading.
LR = hyp_params['lr']


#DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')
# Training images and XML files directory.

TRAIN_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/Final dataset for object detection/train'
# Validation images and XML files directory.
VALID_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/Final dataset for object detection/val'

TEST_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/Final dataset for object detection/test'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'hematoma'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/SSD/outputs_512'