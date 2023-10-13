import torchvision
import torch.nn as nn
from torchvision.models.detection.ssd import (
    SSD, 
    DefaultBoxGenerator,
    SSDHead
)

def create_model(num_classes=2, size=300, nms=0.45):
    # Load the ResNet-50 pretrained model
    model_backbone = torchvision.models.resnet50(pretrained=True)
    
    # Remove the last classification layer of ResNet-50
    # to use it as the feature extractor
    backbone = nn.Sequential(
        model_backbone.conv1,
        model_backbone.bn1,
        model_backbone.relu,
        model_backbone.maxpool,
        model_backbone.layer1,
        model_backbone.layer2,
        model_backbone.layer3,
        model_backbone.layer4
    )
    
    out_channels = [2048, 1024, 512, 256, 256, 256]
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    )
    num_anchors = anchor_generator.num_anchors_per_location()
    head = SSDHead(out_channels, num_anchors, num_classes)
    model = SSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(size, size),
        head=head,
        nms_thresh=nms
    )
    
    return model
