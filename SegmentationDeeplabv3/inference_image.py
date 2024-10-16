import torch
import argparse
import cv2
import os
import time 
from tqdm import tqdm
import torch.nn as nn
from statistics import mean

from datasets import get_test_images, get_test_loader
from metrics import IOUEval
from utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from config import ALL_CLASSES, LABEL_COLORS_LIST
from model import prepare_model
import subprocess
import psutil

def get_gpu_load():
    """
    Returns the current GPU utilization as a string percentage.
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        encoding='utf-8'
    )
    return result.strip()

def test_eval(model, test_dataset, test_dataloader, device, label_colors_list, output_dir):
    model.eval()
    iou_eval = IOUEval(nClasses=len(label_colors_list))  # Update nClasses as per your dataset
    num_batches = int(len(test_dataset)/test_dataloader.batch_size)
    # time.sleep(3)
    # cpu_usage_before = psutil.cpu_percent(interval=None)
    # gpu_load_before = get_gpu_load() if torch.cuda.is_available() else "N/A"
    # cpu_list = []
    # gpu_list = []
    start_time = time.time()
    num_images = 0
    with torch.no_grad():
        prog_bar = tqdm(test_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for i, data in enumerate(prog_bar):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)['out']
            
            # UNCOMMENT FOR METRICS 
            iou_eval.addBatch(outputs.max(1)[1].data, target.data)

            # cpu_usage_after = psutil.cpu_percent(interval=None)
            # gpu_load_after = get_gpu_load() if torch.cuda.is_available() else "N/A"
            # cpu_list.append(cpu_usage_after)
            # gpu_list.append(int(gpu_load_after))

            num_images += data.shape[0]  # Count the number of images processed

        
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
    # UNCOMMENT FOR METRICS 
    overall_acc, per_class_acc, per_class_iou, mIOU, per_class_dice, dice = iou_eval.getMetric()
    return mIOU, dice, per_class_iou, per_class_dice
    #return 0, 0, 0, 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)
ckpt = torch.load(r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\segmentation\Deeplabv3\outputs\best_model_iou.pth')
model.load_state_dict(ckpt['model_state_dict'])
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")
model.eval().to(device)
img_size = 256
batch_size = 1 
test_images, test_masks = get_test_images(
    root_path=r"C:\Users\akumar80\Documents\Avisha Kumar Lab Work\segmentation\Images"   
)
test_dataset, test_dataloader = get_test_loader(
    test_images, 
    test_masks,
    LABEL_COLORS_LIST,
    ALL_CLASSES,
    ALL_CLASSES,
    img_size,
    batch_size
)

# Output directory for overlay images
output_dir = 'outputs/test_inference_results'
os.makedirs(output_dir, exist_ok=True)


# Call the evaluation function
mIOU, dice, per_class_iou, per_class_dice = test_eval(model, test_dataset, test_dataloader, device, LABEL_COLORS_LIST, output_dir)
print(f"Test Set Evaluation - mIOU: {mIOU}, dice: {dice}")
print('per class iu')
print(per_class_iou)
print('per class dice')
print(per_class_dice)

