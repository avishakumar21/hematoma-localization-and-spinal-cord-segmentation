import numpy as np
import cv2
import torch
import glob as glob
import os
from model import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

# Load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/hematoma-localization-and-spinal-cord-segmentation/HematomaDetectionRCNN/outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()
DIR_TEST = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/test'
test_images = glob.glob(f"{DIR_TEST}/*.png")
# Define your IOU threshold
iou_threshold = 0.5

# Initialize variables to store TP, FP, and FN
total_tp = 0
total_fp = 0
total_fn = 0

for i in range(len(test_images)):
    # ... (your existing code for inference)

    # Calculate IOU for each predicted and ground truth box
    for pred_box in boxes:
        for gt_box, gt_label in zip(ground_truth_boxes, ground_truth_labels):
            iou = calculate_iou(pred_box, gt_box)  # Implement calculate_iou function
            if iou >= iou_threshold:
                total_tp += 1
                # You can also keep track of matched ground truth boxes to avoid duplicates
                # Remove the matched gt_box and gt_label from lists
                # This is to ensure each ground truth box is only matched once
            else:
                total_fp += 1

    # Calculate remaining unmatched ground truth boxes as FN
    total_fn += len(ground_truth_boxes)

# Calculate precision and recall
precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)

# Print or use precision and recall as needed
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
