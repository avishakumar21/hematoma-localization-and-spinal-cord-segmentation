import torch
import argparse
import cv2
import os
from statistics import mean 
from tqdm import tqdm
from datasets import get_test_images, get_test_loader
from metrics import IOUEval
import time
from utils import get_segment_labels, draw_segmentation_map, image_overlay
from config import ALL_CLASSES, LABEL_COLORS_LIST
from model import UNet


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
    # cpu_list = []
    # gpu_load_before = get_gpu_load() if torch.cuda.is_available() else "N/A"
    # gpu_list = []
    start_time = time.time()
    num_images = 0
    with torch.no_grad():
        prog_bar = tqdm(test_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for i, data in enumerate(prog_bar):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            # UCOMMENT FOR METRICS
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
model = UNet(num_classes=len(ALL_CLASSES))
ckpt = torch.load(r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\segmentation\UNET\outputs\best_model_iou.pth', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
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




# # Construct the argument parser.
# parser = argparse.ArgumentParser()
# #parser.add_argument('-i', '--input', help='path to input dir')
# parser.add_argument('-i', '--input', default=r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\segmentation\Images\test_images', help='path to input dir')

# parser.add_argument(
#     '--model',
#     default=r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\segmentation\UNET\outputs\best_model_loss.pth',
#     help='path to the model checkpoint'
# )
# parser.add_argument(
#     '--imgsz',
#     default=512,
#     help='image resize resolution',
# )
# args = parser.parse_args()

# out_dir = os.path.join('..', 'outputs', 'inference_results')
# os.makedirs(out_dir, exist_ok=True)

# # Set computation device.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = UNet(num_classes=len(ALL_CLASSES))
# ckpt = torch.load(r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\segmentation\UNET\outputs\best_model_iou.pth', map_location='cpu')
# model.load_state_dict(ckpt['model_state_dict'])
# model.eval().to(device)

# all_image_paths = os.listdir(args.input)
# for i, image_path in enumerate(all_image_paths):
#     print(f"Image {i+1}")
#     # Read the image.
#     image = cv2.imread(os.path.join(args.input, image_path))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     if args.imgsz is not None:
#         image = cv2.resize(image, (int(args.imgsz), int(args.imgsz)))

#     image_copy = image.copy()
#     image_copy = image_copy / 255.0
#     image_tensor = torch.permute(
#         torch.tensor(image_copy, dtype=torch.float32), (2, 0, 1)
#     )
#     # Do forward pass and get the output dictionary.
#     outputs = get_segment_labels(image_tensor, model, device)
#     outputs = outputs
#     segmented_image = draw_segmentation_map(outputs)
    
#     final_image = image_overlay(image, segmented_image)
#     cv2.imshow('Segmented image', final_image)
#     cv2.waitKey(1)
#     cv2.imwrite(os.path.join(out_dir, image_path), final_image)