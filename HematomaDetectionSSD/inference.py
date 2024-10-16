# import numpy as np
# import cv2
# import torch
# import glob as glob
# import os
# import time
# import argparse
# import psutil


# from model import create_model

# from config import (
#     NUM_CLASSES, DEVICE, CLASSES, NUM_WORKERS
# )

# from datasets import (
#     create_test_dataset, 
#     create_test_loader
# )

# from torchmetrics.detection.mean_ap import MeanAveragePrecision

# def test(test_data_loader, model):
#     print('Testing')
#     model.eval()
#     target = []
#     preds = []
#     # initialize tqdm progress bar
#     cpu_usage_before = psutil.cpu_percent(interval=None)

#     # Measure GPU memory usage before inference
#     # if torch.cuda.is_available():
#     #     torch.cuda.synchronize() # Wait for all kernels in all streams on a CUDA device to complete
#     #     start_memory = torch.cuda.memory_allocated()

#     for i, data in enumerate(test_data_loader):
#         images, targets = data
        
#         images = list(image.to(DEVICE) for image in images)
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
#         # for t in targets:
#         #     if t['boxes'].size() == torch.Size([0]):
#         #         boxes = torch.zeros((0, 4), dtype=torch.float32)
#         #         #boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         #         labels = torch.zeros([0], dtype=torch.int64)
#         #         #labels = torch.as_tensor(labels, dtype=torch.int64)
#         #         area = torch.zeros([0], dtype=torch.float32)
#         #         iscrowd = torch.zeros([0], dtype=torch.int64)
#         #         t["boxes"] = boxes.to(DEVICE)
#         #         t["labels"] = labels.to(DEVICE)
#         #         t["area"] = area.to(DEVICE)
#         #         t["iscrowd"] = iscrowd.to(DEVICE)
        
#         with torch.no_grad():
#             outputs = model(images, targets)


#         # For mAP calculation using Torchmetrics.
#         #####################################
#         for i in range(len(images)):
#             true_dict = dict()
#             preds_dict = dict()
#             true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
#             true_dict['labels'] = targets[i]['labels'].detach().cpu()
#             preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
#             preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
#             preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
#             preds.append(preds_dict)
#             target.append(true_dict)
#         #####################################
#     cpu_usage_after = psutil.cpu_percent(interval=None)
#     cpu_usage_diff = cpu_usage_after - cpu_usage_before
#     print(f"CPU usage increase due to inference in test function: {cpu_usage_diff}%")

#     # Measure GPU memory usage after inference
#     # if torch.cuda.is_available():
#     #     torch.cuda.synchronize() # Ensure completion of CUDA operations
#     #     end_memory = torch.cuda.memory_allocated()
#     #     memory_diff = (end_memory - start_memory) / (1024 ** 2) # Convert bytes to megabytes
#     #     print(f"GPU memory used for inference: {memory_diff} MB")
    
#     metric = MeanAveragePrecision()
#     metric.update(preds, target)
#     metric_summary = metric.compute()
#     return metric_summary

import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import psutil
import subprocess
from statistics import mean

from model import create_model
from config import NUM_CLASSES, DEVICE, CLASSES, NUM_WORKERS
from datasets import create_test_dataset, create_test_loader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def get_gpu_load():
    """
    Returns the current GPU utilization as a string percentage.
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        encoding='utf-8'
    )
    return result.strip()

def test(test_data_loader, model):
    print('Testing')
    model.eval()
    target = []
    preds = []
    
    time.sleep(3)
    # Measure CPU and GPU load before inference
    cpu_usage_before = psutil.cpu_percent(interval=None)
    gpu_load_before = get_gpu_load() if torch.cuda.is_available() else "N/A"
    cpu_list = []
    gpu_list = []
    
    for i, data in enumerate(test_data_loader):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)
            cpu_usage_after = psutil.cpu_percent(interval=None)
            gpu_load_after = get_gpu_load() if torch.cuda.is_available() else "N/A"

        cpu_list.append(cpu_usage_after)
        gpu_list.append(int(gpu_load_after))

        # For mAP calculation using Torchmetrics
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
    
    # Measure CPU and GPU load after inference
    cpu_usage_after= mean(cpu_list)

    gpu_load_after = mean(gpu_list)
    
    # Calculate and print usage stats
    cpu_usage_diff = cpu_usage_after - cpu_usage_before
    print(f"CPU usage increase due to inference in test function: {cpu_usage_diff}%")
    print(f"GPU Load before execution: {gpu_load_before}%")
    print(f"GPU Load after execution: {gpu_load_after}%")
    
    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary

# Your main code remains the same.


if __name__ == '__main__':

    np.random.seed(42)

    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        help='path to input image directory',
    )
    parser.add_argument(
        '--imgsz', 
        default=256,
        type=int,
        help='image resize shape'
    )
    parser.add_argument(
        '--threshold',
        default=0.5,
        type=float,
        help='detection threshold'
    )
    args = vars(parser.parse_args())

    os.makedirs('inference_outputs_512/images', exist_ok=True)

    COLORS = [[0, 0, 0], [255, 0, 0]]

    # Load the best model and trained weights.
    model = create_model(num_classes=NUM_CLASSES, size=256)
    checkpoint = torch.load('outputs_512/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    # Directory where all the images are present.
    #DIR_TEST = args['input']
    DIR_TEST = 'C:/Users/akumar80/Documents/Avisha Kumar Lab Work/Final dataset for object detection/test'
    test_images = glob.glob(f"{DIR_TEST}/*.png")
    print(f"Test instances: {len(test_images)}")

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    
    test_dataset = create_test_dataset(DIR_TEST)

    test_loader = create_test_loader(test_dataset, NUM_WORKERS)

    metric_summary = test(test_loader, model)

    print(f"mAP@0.50:0.95: {metric_summary['map']}")
    print(f"mAP@0.50: {metric_summary['map_50']}")   
    print(f"mAP@0.75: {metric_summary['map_75']}")   
    print(f"mAR@1: {metric_summary['mar_1']}")   
    print(f"mAR@10: {metric_summary['mar_10']}")  
    print(f"mAR@large: {metric_summary['mar_large']}")  
    print(f"mAR@medium: {metric_summary['mar_medium']}")  
    print(f"mAR@small: {metric_summary['mar_small']}")

    for i in range(len(test_images)):
        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()

        # UNCOMMMENT THIS 
        if args['imgsz'] is not None:
            image = cv2.resize(image, (args['imgsz'], args['imgsz']))
        print(image.shape)
        # BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Make the pixel range between 0 and 1.
        image /= 255.0
        # Bring color channels to front (H, W, C) => (C, H, W).
        image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # Convert to tensor.
        image_input = torch.tensor(image_input, dtype=torch.float).cuda()
        # Add batch dimension.
        image_input = torch.unsqueeze(image_input, 0)
        start_time = time.time()
        # Predictions
        with torch.no_grad():
            outputs = model(image_input.to(DEVICE))
        end_time = time.time()

        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Total FPS till current frame.
        total_fps += fps
        frame_count += 1

        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # Filter out boxes according to `detection_threshold`.
            boxes = boxes[scores >= args['threshold']].astype(np.int32)
            draw_boxes = boxes.copy()
            # Get all the predicited class names.
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            
            # Draw the bounding boxes and write the class name on top of it.
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = COLORS[CLASSES.index(class_name)]
                # Recale boxes.
                xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
                ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
                xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
                ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
                cv2.rectangle(orig_image,
                            (xmin, ymin),
                            (xmax, ymax),
                            color[::-1], 
                            3)
                cv2.putText(orig_image, 
                            class_name, 
                            (xmin, ymin-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            color[::-1], 
                            2, 
                            lineType=cv2.LINE_AA)

            #cv2.imshow('Prediction', orig_image)
            #cv2.waitKey(1)
            cv2.imwrite(f"C:/Users/akumar80/Documents/Avisha Kumar Lab Work/SSD/inference_outputs_512/images/{image_name}.png", orig_image)
        # print(f"Image {i+1} done...")
        # print('-'*50)

    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()
    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")