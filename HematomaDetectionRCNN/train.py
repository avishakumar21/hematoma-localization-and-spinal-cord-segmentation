from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS, CLASSES
)

from engine import evaluate
from model import create_model
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader, CustomDataset
)

import torch
import glob as glob
import matplotlib.pyplot as plt
import time
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from config import (
    NUM_CLASSES, DEVICE, CLASSES
)
plt.style.use('ggplot')


# function for running training iterations
def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list
    model.train()

    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets] # this tensor is passed through TO DEVICE (might be making 0 values to null -- causes dimension error)
        for t in targets:
            if t['boxes'].size() == torch.Size([0]):
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                #boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.zeros([0], dtype=torch.int64)
                #labels = torch.as_tensor(labels, dtype=torch.int64)
                area = torch.zeros([0], dtype=torch.float32)
                iscrowd = torch.zeros([0], dtype=torch.int64)
                t["boxes"] = boxes.to(DEVICE)
                t["labels"] = labels.to(DEVICE)
                t["area"] = area.to(DEVICE)
                t["iscrowd"] = iscrowd.to(DEVICE)
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        

        losses.backward()
        optimizer.step()
        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    

    return train_loss_list



    # function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list

    model.eval() 
    # Initialize empty lists to store true labels and predicted labels
    target = []
    preds = []
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        for t in targets:
            if t['boxes'].size() == torch.Size([0]):
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                #boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.zeros([0], dtype=torch.int64)
                #labels = torch.as_tensor(labels, dtype=torch.int64)
                area = torch.zeros([0], dtype=torch.float32)
                iscrowd = torch.zeros([0], dtype=torch.int64)
                t["boxes"] = boxes.to(DEVICE)
                t["labels"] = labels.to(DEVICE)
                t["area"] = area.to(DEVICE)
                t["iscrowd"] = iscrowd.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
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
        #####################################

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary




  


if __name__ == '__main__':
    print(DEVICE)
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")
    # initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []


    # name to save the trained model with
    MODEL_NAME = 'model'
    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image
        show_tranformed_image(train_loader)
    # initialize SaveBestModel class
    save_best_model = SaveBestModel()

    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)

        metric_summary = validate(valid_loader, model)

        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")  
        print(f"Epoch #{epoch+1} mAP@0.50:0.95: {metric_summary['map']}")
        print(f"Epoch #{epoch+1} mAP@0.50: {metric_summary['map_50']}")   
        print(f"Epoch #{epoch+1} mAP@0.75: {metric_summary['map_75']}")   
        print(f"Epoch #{epoch+1} mAR@1: {metric_summary['mar_1']}")   
        print(f"Epoch #{epoch+1} mAR@10: {metric_summary['mar_10']}")  
        print(f"Epoch #{epoch+1} mAR@large: {metric_summary['mar_large']}")  
        print(f"Epoch #{epoch+1} mAR@medium: {metric_summary['mar_medium']}")  
        print(f"Epoch #{epoch+1} mAR@small: {metric_summary['mar_small']}")

        # print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        # print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")
        # save the best model till now if we have the least loss in the...
        # ... current epoch
        # save_best_model(
        #     val_loss_hist.value, epoch, model, optimizer
        # )

                # save the best model till now.
        save_best_model(
            model, float(metric_summary['map']), epoch, 'outputs'
        )
        # Save the current epoch model.
        save_model(epoch, model, optimizer)

        # save loss plot
        # save_loss_plot(OUT_DIR, train_loss, val_loss)

        # Append the epoch's average training loss to the list
        # train_loss_list_epochs.append(train_loss_hist.value)
        #val_loss_list_epochs.append(val_loss_hist.value)

        #evaluate(model, valid_loader, device=DEVICE)


        # sleep for 5 seconds after each epoch
        time.sleep(5)
        


    # # Plot train_loss_list_epochs vs epochs
    # plt.figure(figsize=(8, 6))
    # print(range(NUM_EPOCHS))
    # print(train_loss_list_epochs)
    # plt.plot(range(NUM_EPOCHS), train_loss_list_epochs, label='Train Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss per Epoch')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # # Plot val_loss vs epochs
    # plt.figure(figsize=(8, 6))
    # print(range(NUM_EPOCHS))
    # print(val_loss_list_epochs)
    # plt.plot(range(NUM_EPOCHS), val_loss_list_epochs, label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Validation Loss per Epoch')
    # plt.legend()
    # plt.grid(True)
    # plt.show()



