import torch
import os
import argparse
import nni 
from datasets import get_images, get_dataset, get_data_loaders
from model import segformer_model
from config import ALL_CLASSES, LABEL_COLORS_LIST
from transformers import SegformerFeatureExtractor
from engine import train, validate
from utils import save_model, SaveBestModel, save_plots, SaveBestModelIOU
from torch.optim.lr_scheduler import MultiStepLR

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

hyp_params = {'batch': 4, 'lr': 0.0009722979689763296, 'epochs': 75}
#hyp_params = {'batch': 4, 'lr': 0.001, 'epochs': 75}


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
    default=hyp_params['lr'],
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
    default=[256, 256],
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--scheduler',
    action='store_true',
)
parser.add_argument(
    '--scheduler-epochs',
    dest='scheduler_epochs',
    default=[30],
    nargs='+',
    type=int
)
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    # Create a directory with the model name for outputs.
    out_dir = os.path.join('outputs')
    out_dir_valid_preds = os.path.join(out_dir, 'valid_preds')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = segformer_model(classes=ALL_CLASSES).to(device)
    #print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_images, train_masks, valid_images, valid_masks = get_images(
        root_path=r"C:\Users\akumar80\Documents\Avisha Kumar Lab Work\segmentation\Images"   
    )

    feature_extractor = SegformerFeatureExtractor(size=(args.imgsz[1], args.imgsz[0]))

    train_dataset, valid_dataset = get_dataset(
        train_images, 
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        ALL_CLASSES,
        LABEL_COLORS_LIST,
        img_size=args.imgsz,
        feature_extractor=feature_extractor
    )

    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, 
        valid_dataset,
        args.batch
    )

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()
    save_best_iou = SaveBestModelIOU()
    # LR Scheduler.
    scheduler = MultiStepLR(
        optimizer, milestones=args.scheduler_epochs, gamma=0.1, verbose=True
    )

    train_loss, train_pix_acc, train_miou = [], [], []
    valid_loss, valid_pix_acc, valid_miou = [], [], []
    
    for epoch in range (args.epochs):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_pixacc, train_epoch_miou = train(
            model,
            train_dataloader,
            device,
            optimizer,
            ALL_CLASSES
        )
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_miou = validate(
            model,
            valid_dataloader,
            device,
            ALL_CLASSES,
            LABEL_COLORS_LIST,
            epoch,
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
        #if args.scheduler:
        scheduler.step()
        print('-' * 50)

    # # Save the loss and accuracy plots.
    save_plots(
        train_pix_acc, valid_pix_acc, 
        train_loss, valid_loss,
        train_miou, valid_miou, 
        out_dir
    )
    # Save final model.
    save_model(model, out_dir, name='final_model')
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