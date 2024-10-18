import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset, CustomGenerator
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils import DiceLoss, IOULoss

from torchvision import transforms

volume_path = '../SegmentationData/full_dataset/test_vol_h5_full'

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default=volume_path, help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=10, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=128, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
# parser.add_argument('--n_skip', type=int, default=0, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol_full", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        
        # Log Mean IOU
        logging.info('idx %d case %s mean_iou %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[2]))

    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_miou = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    # Log Mean IOU
    logging.info('Testing performance in best val model: mean_iou : %f' % (mean_miou))
    return "Testing Finished!"


    ################################################
    # SAME AS TRAINER.PY CODE BELOW
    #################################################
    # db_test = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="train_npz_small",
    #                            transform=transforms.Compose([CustomGenerator(output_size=[args.img_size, args.img_size])]))

    # testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # model.eval()
    # dice_loss = DiceLoss(args.num_classes)
    # iou_loss = IOULoss(args.num_classes)
    # # optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    # # writer = SummaryWriter(snapshot_path + '/log')
    # iter_num = 0
    # max_epoch = args.max_epochs
    # max_iterations = args.max_epochs * len(testloader)  # max_epoch = max_iterations // len(trainloader) + 1
    # logging.info("{} iterations per epoch. {} max iterations ".format(len(testloader), max_iterations))
    # best_performance = 0.0
    # iterator = tqdm(range(max_epoch), ncols=70)
    # with torch.no_grad():
    #     for i_batch, sampled_batch in enumerate(testloader):
    #         image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
    #         image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
    #         outputs = model(image_batch)
    #             # loss_ce = ce_loss(outputs, label_batch[:].long())
    #             # loss_dice = dice_loss(outputs, label_batch, softmax=True)
    #         loss_iou = iou_loss(outputs, label_batch, softmax=True)
    #         iou = 1 - loss_iou
    #         # iter_num = iter_num + 1
    #             # writer.add_scalar('info/lr', lr_, iter_num)
    #             # writer.add_scalar('info/total_loss', loss, iter_num)
    #             # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
                

    #             # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
    #         logging.info('iou: %f' % (iou.item()))

    # metric_list = metric_list / len(db_test)
    # for i in range(1, args.num_classes):
    #     # logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))

    #     # Log Mean IOU
    #     logging.info('Mean class %d mean_iou %f' % (i, metric_list[i-1][2]))

    ################################################
    # SAME AS TRAINER.PY CODE ABOVE
    #################################################
    




if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 10,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = './predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


