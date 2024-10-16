import torch
import torch.nn as nn
import cv2
import numpy as np
from statistics import mean 
import os
from datasets import get_test_images, get_test_loader
from config import LABEL_COLORS_LIST, ALL_CLASSES
from metrics import IOUEval
from tqdm import tqdm
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from utils import draw_segmentation_map, image_overlay, predict
import subprocess
import psutil
import time 

def get_gpu_load():
    """
    Returns the current GPU utilization as a string percentage.
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        encoding='utf-8'
    )
    return result.strip()

def test_eval(model, test_dataloader, device, label_colors_list, output_dir):
    model.eval()
    iou_eval = IOUEval(nClasses=len(label_colors_list))  # Update nClasses as per your dataset
    #time.sleep(3)
    # cpu_usage_before = psutil.cpu_percent(interval=None)
    # cpu_list = []
    # gpu_load_before = get_gpu_load() if torch.cuda.is_available() else "N/A"
    # gpu_list = []
    start_time = time.time()
    num_images = 0
    image_metrics = []  # List to store the metrics for each image

    with torch.no_grad():
        for i,data in enumerate(tqdm(test_dataloader, desc="Test Batches")):
                pixel_values, target = data['pixel_values'].to(device), data['labels'].to(device)
                outputs = model(pixel_values=pixel_values)

                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(
                    logits, size=target.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
                )

                predictions = upsampled_logits.argmax(dim=1)  # Convert logits to predictions
                iou_eval.addBatch(predictions, target)

                hist = iou_eval.compute_hist(predictions.cpu().numpy().flatten(), target.cpu().numpy().flatten())
                epsilon = 0.00000001
                per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
                per_class_dice = 2 * np.diag(hist) / (2 * np.diag(hist) + hist.sum(axis=0) - np.diag(hist) + hist.sum(axis=1) - np.diag(hist) + epsilon)
                mIOU = np.nanmean(per_class_iu)
                dice = np.nanmean(per_class_dice)
                
                image_metrics.append((i, mIOU, dice))

                num_images += pixel_values.shape[0]
                
                #ADD THIS BACK FOR METRICS
                # cpu_usage_after = psutil.cpu_percent(interval=None)
                # gpu_load_after = get_gpu_load() if torch.cuda.is_available() else "N/A"
                # cpu_list.append(cpu_usage_after)
                # gpu_list.append(int(gpu_load_after))
                num_images += pixel_values.shape[0]  # Count the number of images processed

    # cpu_avg = mean(cpu_list)
    # cpu_usage_diff = cpu_avg - cpu_usage_before
    # gpu_avg = mean(gpu_list)
    # print(f"CPU usage increase due to inference in test function: {cpu_usage_diff}%")

    # print(f"GPU Load before execution: {gpu_load_before}%")
    # print(f"GPU Load after execution: {gpu_avg}%")
    
    # Stop timing and calculate FPS
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_images / total_time  # Calculate frames per second
    print(f"FPS: {fps}")
          
    # Compute final IOU metrics

    image_metrics_sorted_by_miou = sorted(image_metrics, key=lambda x: x[1])
    image_metrics_sorted_by_dice = sorted(image_metrics, key=lambda x: x[2])

    print("Images with the 20 lowest mIOU:")
    for img in image_metrics_sorted_by_miou[:20]:
        print(f"Image index: {img[0]}, mIOU: {img[1]}")

    print("Images with the 20 lowest dice:")
    for img in image_metrics_sorted_by_dice[:20]:
        print(f"Image index: {img[0]}, Dice: {img[2]}")

    # ADD THIS BACK FOR METRICS 
    overall_acc, per_class_acc, per_class_iou, mIOU, per_class_dice, dice = iou_eval.getMetric()

    return mIOU, dice, per_class_iou, per_class_dice
    #return 0, 0, 0, 0


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
extractor = SegformerFeatureExtractor(size=(256, 256))
model = SegformerForSemanticSegmentation.from_pretrained('outputs/model_iou')
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")
model.to(device).eval()
img_size = [256, 256]
batch_size = 1 
test_images, test_masks = get_test_images(
    root_path=r"C:\Users\akumar80\Documents\Avisha Kumar Lab Work\segmentation\Images"   
)

test_dataloader = get_test_loader(
    test_images, 
    test_masks,
    LABEL_COLORS_LIST,
    ALL_CLASSES,
    ALL_CLASSES,
    img_size,
    extractor,
    batch_size
)

# Output directory for overlay images
output_dir = 'outputs/test_inference_results'
os.makedirs(output_dir, exist_ok=True)


# Call the evaluation function
mIOU, dice, per_class_iou, per_class_dice= test_eval(model, test_dataloader, device, LABEL_COLORS_LIST, output_dir)
print(f"Test Set Evaluation - mIOU: {mIOU}, dice: {dice}")
print('per class iu')
print(per_class_iou)
print('per class dice')
print(per_class_dice)
