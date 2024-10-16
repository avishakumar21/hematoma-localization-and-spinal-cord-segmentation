import torch
import torch.nn as nn
import os
import argparse

from datasets import get_images, get_dataset, get_data_loaders
from engine import train, validate
from model import UNet
from config import ALL_CLASSES, LABEL_COLORS_LIST
from utils import save_model, SaveBestModel, save_plots, SaveBestModelIOU, EarlyStopping
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nni 

# Dice Loss Function
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        # Ensure inputs and targets are on the same device
        device = inputs.device

        # One-hot encode targets to have the same shape as inputs
        targets = torch.eye(inputs.shape[1], device=device)[targets.squeeze(1)]
        targets = targets.permute(0, 3, 1, 2).contiguous()

        inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)

        intersection = (inputs * targets).sum(2)
        union = inputs.sum(2) + targets.sum(2)
        dice = (2.*intersection + smooth) / (union + smooth)  
        return 1 - dice.mean()



seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

hyp_params = {'batch': 8, 'lr': 0.0030113764281910604, 'epochs': 50}

# optimized_params = nni.get_next_parameter()
# hyp_params.update(optimized_params)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--epochs',
    default=hyp_params['epochs'],
    help='number of epochs to train for',
    type=int
)
parser.add_argument(
    '--lr',
    default=hyp_params['lr'], # 0.01 - 0.0001
    help='learning rate for optimizer',
    type=float
)
parser.add_argument(
    '--batch',
    default=hyp_params['batch'],
    help='batch size for data loader',
    type=int
)
parser.add_argument(
    '--imgsz', 
    default=256,
    type=int
)
parser.add_argument(
    '--scheduler',
    action='store_true',
)
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    # Create a directory with the model name for outputs.
    out_dir = os.path.join('..', 'outputs')
    out_dir_valid_preds = os.path.join('..', 'outputs', 'valid_preds')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(num_classes=len(ALL_CLASSES)).to(device)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()  
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #criterion = DiceLoss()  # Using Dice Loss

    train_images, train_masks, valid_images, valid_masks = get_images(
        root_path=r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\segmentation\Images'   
    )

    classes_to_train = ALL_CLASSES

    train_dataset, valid_dataset = get_dataset(
        train_images, 
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        classes_to_train,
        LABEL_COLORS_LIST,
        img_size=args.imgsz
    )

    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, valid_dataset, batch_size=args.batch
    )

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()
    save_best_iou = SaveBestModelIOU()

    # LR Scheduler.
    scheduler = MultiStepLR(
        optimizer, milestones=[60], gamma=0.1, verbose=True
    )
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)
    #early_stopping = EarlyStopping(patience=10, verbose=True)  # Adjust patience as needed


    EPOCHS = args.epochs
    train_loss, train_pix_acc, train_miou = [], [], []
    valid_loss, valid_pix_acc, valid_miou = [], [], []
    for epoch in range (EPOCHS):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_pixacc, train_epoch_miou = train(
            model,
            train_dataset,
            train_dataloader,
            device,
            optimizer,
            criterion,
            classes_to_train
        )
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_miou = validate(
            model,
            valid_dataset,
            valid_dataloader,
            device,
            criterion,
            classes_to_train,
            LABEL_COLORS_LIST,
            epoch,
            ALL_CLASSES,
            save_dir=out_dir_valid_preds
        )
        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc)
        train_miou.append(train_epoch_miou)
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc)
        valid_miou.append(valid_epoch_miou)

        # nni.report_intermediate_result(float(valid_epoch_miou))
        # if epoch == hyp_params['epochs'] - 1:
        #     nni.report_final_result(valid_epoch_miou)

        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, name='model_loss'
        )
        save_best_iou(
            valid_epoch_miou, epoch, model, out_dir, name='model_iou'
        )

        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f},",
            f"Train Epoch PixAcc: {train_epoch_pixacc:.4f},",
            f"Train Epoch mIOU: {train_epoch_miou:4f}"
        )
        print(
            f"Valid Epoch Loss: {valid_epoch_loss:.4f},", 
            f"Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}",
            f"Valid Epoch mIOU: {valid_epoch_miou:4f}"
        )
        # if args.scheduler:
        scheduler.step()
        # scheduler.step(valid_epoch_loss)
        # early_stopping(valid_epoch_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        print('-' * 50)
 

    save_model(EPOCHS, model, optimizer, criterion, out_dir, name='model')
    # Save the loss and accuracy plots.
    save_plots(
        train_pix_acc, valid_pix_acc, 
        train_loss, valid_loss,
        train_miou, valid_miou, 
        out_dir
    )
    print('TRAINING COMPLETE')
    print('train miou: ')
    print(train_miou)
    print('valid miou: ')
    print(valid_miou)
    print('train pix acc')
    print(train_pix_acc)
    print('valid pix acc')
    print(valid_pix_acc)
    print('train loss')
    print(train_loss)
    print('valid loss')
    print(valid_loss)