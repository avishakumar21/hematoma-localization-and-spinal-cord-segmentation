import glob
import albumentations as A
import cv2

from utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def get_images(root_path):
    train_images = glob.glob(f"{root_path}/train_images/*")
    train_images.sort()
    train_masks = glob.glob(f"{root_path}/train_masks/*")
    train_masks.sort()
    valid_images = glob.glob(f"{root_path}/val_images/*") 
    valid_images.sort()
    valid_masks = glob.glob(f"{root_path}/val_masks/*")
    valid_masks.sort()

    return train_images, train_masks, valid_images, valid_masks

def get_test_images(root_path):
    test_images = glob.glob(f"{root_path}/human_images/*")
    test_images.sort()
    test_masks = glob.glob(f"{root_path}/human_masks/*")
    test_masks.sort()

    return test_images, test_masks


def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    # train_image_transform = A.Compose([
    #     A.Resize(img_size[1], img_size[0], always_apply=True),
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.2),
    #     A.Rotate(limit=25),
    # ], is_check_shapes=False)
    # return train_image_transform
    train_image_transform = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.Rotate(limit=25),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.2),
    ], is_check_shapes=False)
    return train_image_transform

def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    valid_image_transform = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
    ], is_check_shapes=False)
    return valid_image_transform

class SegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        label_colors_list,
        classes_to_train,
        all_classes,
        feature_extractor
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')

        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image'].astype('uint8')
        mask = transformed['mask']
        
        # Get 2D label mask.
        mask = get_label_mask(mask, self.class_values, self.label_colors_list).astype('uint8')
        mask = Image.fromarray(mask)
               
        encoded_inputs = self.feature_extractor(
            Image.fromarray(image), 
            mask,
            return_tensors='pt'
        )
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs

def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size,
    feature_extractor
):
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes, 
        feature_extractor
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        label_colors_list,
        classes_to_train,
        all_classes,
        feature_extractor
    )
    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        num_workers=0,
        shuffle=True
    )
    valid_data_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        num_workers=0,
        shuffle=False
    )

    return train_data_loader, valid_data_loader

def get_test_loader(test_image_paths, test_mask_paths, label_colors_list, classes_to_train, all_classes, img_size, feature_extractor, batch_size):
    # Reuse the valid_transforms for test dataset
    test_tfms = valid_transforms(img_size)
    test_dataset = SegmentationDataset(
        test_image_paths,
        test_mask_paths,
        test_tfms,
        label_colors_list,
        classes_to_train,
        all_classes,
        feature_extractor
    )
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        shuffle=False  
    )
    
    return test_data_loader
