from transformers import (
    SegformerFeatureExtractor, 
    SegformerForSemanticSegmentation
)
from config import VIS_LABEL_MAP as LABEL_COLORS_LIST
from utils import (
    draw_segmentation_map, 
    image_overlay,
    predict
)

import argparse
import cv2
import os
import glob
import numpy as np 
import torch

# Assuming LABEL_COLORS_LIST is accessible globally
# Map RGB values to class indices
def rgb_to_class_index(image, label_colors_list):
    """
    Convert an RGB image to a 2D array where each pixel's value represents the class index.
    
    Args:
    - image (numpy.ndarray): The RGB image as a numpy array.
    - label_colors_list (list): A list of RGB values that correspond to class indices.
    
    Returns:
    - numpy.ndarray: A 2D array with the same width and height as the input image, 
                     where each pixel's value represents the class index.
    """
    class_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for idx, color in enumerate(label_colors_list):
        # Find pixels matching the current color and set their value to the class index
        class_map[(image == color).all(axis=2)] = idx
    return class_map

def load_ground_truth_labels(image_path, labels_dir, label_colors_list):
    """
    Load ground truth RGB labels and convert them to class indices without resizing.
    
    Args:
    - image_path (str): Path to the input image.
    - labels_dir (str): Directory where label images are stored.
    - label_colors_list (list): List of RGB values for class indices mapping.
    
    Returns:
    - numpy.ndarray: Ground truth labels as a 2D array of class indices.
    """
    # Construct the path to the corresponding label image
    image_filename = os.path.basename(image_path)
    label_image_path = os.path.join(labels_dir, image_filename)
    
    # Load the RGB label image
    label_image = cv2.imread(label_image_path, cv2.IMREAD_COLOR)
    
    # Convert RGB label image to class indices
    label_class_indices = rgb_to_class_index(label_image, label_colors_list)
    
    return label_class_indices



class IOUEval:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
        self.per_class_iu = np.zeros(self.nClasses, dtype=np.float32)
        self.mIOU = 0
        self.batchCount = 1

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.nClasses)
        return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses ** 2).reshape(self.nClasses, self.nClasses)

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        # Check if 'predict' is a PyTorch tensor and convert it to a numpy array if so
        if torch.is_tensor(predict):
            predict = predict.cpu().numpy()
        # Similarly for 'gth'
        if torch.is_tensor(gth):
            gth = gth.cpu().numpy()
        
        # Now both 'predict' and 'gth' should be numpy arrays, and we can flatten them
        predict = predict.flatten()
        gth = gth.flatten()

        epsilon = 0.00000001
        hist = self.compute_hist(predict, gth)
        overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
        mIou = np.nanmean(per_class_iu)

        self.overall_acc += overall_acc
        self.per_class_acc += per_class_acc
        self.per_class_iu += per_class_iu
        self.mIOU += mIou
        self.batchCount += 1


    def getMetric(self):
        overall_acc = self.overall_acc/self.batchCount
        per_class_acc = self.per_class_acc / self.batchCount
        per_class_iu = self.per_class_iu / self.batchCount
        mIOU = self.mIOU / self.batchCount

        return overall_acc, per_class_acc, per_class_iu, mIOU


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input image directory',
    default=r"C:\Users\akumar80\Documents\Avisha Kumar Lab Work\segmentation\Images\test_images"   
)
parser.add_argument(
    '--device',
    default='cuda:0',
    help='compute device, cpu or cuda'
)
parser.add_argument(
    '--imgsz', 
    default=[690, 275],
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--model',
    default='outputs/model_iou'
)
args = parser.parse_args()

out_dir = 'outputs/inference_results_image'
os.makedirs(out_dir, exist_ok=True)

extractor = SegformerFeatureExtractor()
model = SegformerForSemanticSegmentation.from_pretrained(args.model)
model.to(args.device).eval()

iou_eval = IOUEval(nClasses=10)  # Specify the correct number of classes
label_folder = r"C:\Users\akumar80\Documents\Avisha Kumar Lab Work\segmentation\Images\test_masks"

image_paths = glob.glob(os.path.join(args.input, '*'))
for image_path in image_paths:
    image = cv2.imread(image_path)
    # if args.imgsz is not None:
    #     image = cv2.resize(image, (args.imgsz[0], args.imgsz[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get labels.
    labels = predict(model, extractor, image, args.device)

    # Convert model output to class indices if necessary
    if labels.dim() > 2:  # Assuming labels are [C, H, W] or [N, C, H, W], with C being channels
        labels = labels.argmax(dim=1)  # Take argmax across channel dimension

    
    # Load or otherwise obtain the true labels for the current image
    true_labels = load_ground_truth_labels(image_path, label_folder, LABEL_COLORS_LIST)  # need to implement this function
    # Before calling addBatch
    if labels.shape != true_labels.shape:
        raise ValueError(f"Shape mismatch between predictions ({labels.shape}) and ground truth ({true_labels.shape})")
    # Update IOU statistics
    iou_eval.addBatch(labels, true_labels)
   
    # Get segmentation map.
    seg_map = draw_segmentation_map(
        labels.cpu(), LABEL_COLORS_LIST
    )
    outputs = image_overlay(image, seg_map)
    #cv2.imshow('Image', outputs)
    #cv2.waitKey(0)
    
    # Save path.
    image_name = image_path.split(os.path.sep)[-1]
    save_path = os.path.join(
        out_dir, '_'+image_name
    )
    cv2.imwrite(save_path, outputs)

overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval.getMetric()
print('mIOU')
print(mIOU) 