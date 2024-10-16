import torch
import torch.nn as nn

from tqdm import tqdm
from utils import draw_translucent_seg_maps
from metrics import IOUEval

def train(
    model,
    train_dataloader,
    device,
    optimizer,
    classes_to_train
):
    print('Training')
    model.train()
    train_running_loss = 0.0
    prog_bar = tqdm(
        train_dataloader, 
        total=len(train_dataloader), 
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    counter = 0 # to keep track of batch counter
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    for i, data in enumerate(prog_bar):
        counter += 1
        pixel_values, target = data['pixel_values'].to(device), data['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=target)

        ##### BATCH-WISE LOSS #####
        loss = outputs.loss
        train_running_loss += loss.item()
        ###########################
 
        ##### BACKPROPAGATION AND PARAMETER UPDATION #####
        loss.backward()
        optimizer.step()
        ##################################################

        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits, size=target.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        iou_eval.addBatch(upsampled_logits.max(1)[1].data, target.data)
        
    ##### PER EPOCH LOSS #####
    train_loss = train_running_loss / counter
    ##########################
    overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval.getMetric()
    return train_loss, overall_acc, mIOU

def validate(
    model,
    valid_dataloader,
    device,
    classes_to_train,
    label_colors_list,
    epoch,
    save_dir
):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    with torch.no_grad():
        prog_bar = tqdm(
            valid_dataloader, 
            total=(len(valid_dataloader)), 
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        counter = 0 # To keep track of batch counter.
        for i, data in enumerate(prog_bar):
            counter += 1
            pixel_values, target = data['pixel_values'].to(device), data['labels'].to(device)
            outputs = model(pixel_values=pixel_values, labels=target)

            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits, size=target.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            # Save the validation segmentation maps.
            if i == 1:
                draw_translucent_seg_maps(
                    pixel_values, 
                    upsampled_logits, 
                    epoch, 
                    i, 
                    save_dir, 
                    label_colors_list,
                )

            ##### BATCH-WISE LOSS #####
            loss = outputs.loss
            valid_running_loss += loss.item()
            ###########################

            iou_eval.addBatch(upsampled_logits.max(1)[1].data, target.data)
        
    ##### PER EPOCH LOSS #####
    valid_loss = valid_running_loss / counter
    ##########################
    overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval.getMetric()
    return valid_loss, overall_acc, mIOU