import numpy as np
import torch
import os
import openslide
import glob
import random
import pandas as pd
from os.path import exists
from skimage import color
import copy
from skimage.color import rgb2hed
from skimage.color import hed2rgb
from tqdm import tqdm
from torch.utils.data import Dataset
from models.randaugment import RandAugment
from torchvision import transforms
from PIL import Image
from albumentations import Compose, Rotate, CenterCrop, HorizontalFlip, RandomScale, Flip, Resize, ShiftScaleRotate, \
    RandomCrop, IAAAdditiveGaussianNoise, ElasticTransform, HueSaturationValue, LongestMaxSize, RandomBrightnessContrast, Blur


############# List of data augmentations for fine-tuning ########################

def HSV(img):
    transform = Compose([HueSaturationValue(hue_shift_limit=(-0.1, 0.1), sat_shift_limit=(-1, 1))])
    Aug_img = transform(image=img)
    return Aug_img

def Noise(img):
    transform = Compose([IAAAdditiveGaussianNoise(loc=0, scale=(0 * 255, 0.1 * 255))])
    Aug_img = transform(image=img)
    return Aug_img

def Scale_Resize_Crop(img):
    transform = Compose([Rotate(limit=(-90, 90), interpolation=2), RandomScale(scale_limit=(0.8, 1.2), interpolation=2), Resize(img.shape[1] + 20, img.shape[1] + 20, interpolation=2),
                         RandomCrop(img.shape[1], img.shape[1])])
    Aug_img = transform(image=img)
    return Aug_img

def Shift_Scale_Rotate(img):
    transform = Compose([HorizontalFlip(), ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, interpolation=2),
                         RandomCrop(img.shape[1], img.shape[1])])
    Aug_img = transform(image=img)
    return Aug_img

def Blur_img(img):
    transform = Compose([Blur(blur_limit=(3, 7))])
    Aug_img = transform(image=img)
    return Aug_img

def Brightness_Contrast(img):
    transform = Compose([RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2))])
    Aug_img = transform(image=img)
    return Aug_img

def Rotate_Crop(img):
    transform = Compose([Rotate(limit=(-90, 90), interpolation=2), CenterCrop(img.shape[1], img.shape[1])])
    Aug_img = transform(image=img)
    return Aug_img

def augment_pool():
    augs = [HSV, Noise, Scale_Resize_Crop, Shift_Scale_Rotate, Blur_img, Brightness_Contrast, Rotate_Crop]
    return augs
###########


#################################################################
" Supervised fine-tuning on TB Lymph node "

class DatasetTBLN_Supervised_train:

    def __init__(self, dataset_path, image_size):

        """
        TBLN dataset class wrapper (train with augmentation)
        """

        self.image_size = image_size

        # Resize images
        self.transform1 = Compose([Resize(image_size, image_size, interpolation=2)])

        # Data augmentations
        self.augment_pool = augment_pool()

        self.datalist = []
        cls_paths = glob.glob('{}/*/'.format(dataset_path))
        with tqdm(enumerate(sorted(cls_paths)), disable=True) as t:
            for wj, cls_path in t:
                cls_id = str(os.path.split(os.path.dirname(cls_path))[-1])
                cls_id = os.path.basename(cls_id[0])
                patch_pths = glob.glob('{}/*'.format(cls_path))
                for pth in patch_pths:
                    self.datalist.append((pth, cls_id))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        img = Image.open(self.datalist[index][0]).convert('RGB')

        # label assignment
        label = int(self.datalist[index][1])

        #################
        # Convert PIL image to numpy array
        img = np.array(img)

        # First image
        img = self.transform1(image=img)
        img = Image.fromarray(img['image'])
        img = np.array(img)

        #####
        rad_aug_idx = torch.randperm(2)   # randomize 2 out of total 7 augmentations
        ops = [self.augment_pool[i] for i in rad_aug_idx]
        for op in ops:
            img = op(img)
            if isinstance(img, dict):
                img = Image.fromarray(img['image'])
                img = np.array(img)
            else:
                img = img

        # Numpy to torch
        img = torch.from_numpy(img)
        label = np.array(label)
        label = torch.from_numpy(label)

        # Change Tensor Dimension to N x C x H x W
        img = img.permute(2, 0, 1)

        return img, label

##########
class DatasetTBLN_eval:

    def __init__(self, dataset_path, image_size):

        """
        TBLN dataset class wrapper (val)
        """

        self.image_size = image_size

        # Resize images
        self.transform1 = Compose([Resize(image_size, image_size, interpolation=2)])

        self.datalist = []
        cls_paths = glob.glob('{}/*/'.format(dataset_path))
        with tqdm(enumerate(sorted(cls_paths)), disable=True) as t:
            for wj, cls_path in t:
                cls_id = str(os.path.split(os.path.dirname(cls_path))[-1])
                cls_id = os.path.basename(cls_id[0])
                patch_pths = glob.glob('{}/*'.format(cls_path))
                for pth in patch_pths:
                    self.datalist.append((pth, cls_id))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        img = Image.open(self.datalist[index][0]).convert('RGB')

        # label assignment
        label = int(self.datalist[index][1])

        # Convert PIL image to numpy array
        img = np.array(img)

        img1 = self.transform1(image=img)

        # Convert numpy array to PIL Image
        img1 = Image.fromarray(img1['image'])
        img1 = np.array(img1)
        img = torch.from_numpy(img1)

        label = np.array(label)
        label = torch.from_numpy(label)

        # Change Tensor Dimension to N x C x H x W
        img = img.permute(2, 0, 1)

        return img, label

#############


######################################################################################################################

####### MHIST dataset

### MHIST train loader
class DatasetMHIST_train:

    def __init__(self, dataset_path, annot_path, image_size):

        """
        MHIST dataset class wrapper (train with augmentation)
        """

        self.image_size = image_size

        # Resize images
        self.transform1 = Compose([Resize(image_size, image_size, interpolation=2)])

        # Data augmentations
        self.transform4 = Compose([Rotate(limit=(-90, 90), interpolation=2), CenterCrop(image_size, image_size)])
        self.transform5 = Compose([Rotate(limit=(-90, 90), interpolation=2), RandomScale(scale_limit=(0.8, 1.2), interpolation=2),
                                   Resize(image_size + 20, image_size + 20, interpolation=2), RandomCrop(image_size, image_size)])

        # GT annotation
        GT = pd.read_csv(annot_path, header=None)

        self.datalist = []
        img_paths = glob.glob('{}/*.png'.format(dataset_path))
        with tqdm(enumerate(sorted(img_paths)), disable=True) as t:
            for wj, img_path in t:
                head, tail = os.path.split(img_path)
                img_id = tail  # Get image_id

                # check if it belongs to train/val set
                set = GT.loc[GT[0] == img_id][3]
                label = GT.loc[GT[0] == img_id][1]

                # Add only train/test to the corresponding set
                if set.iloc[0] == 'train':
                    if label.iloc[0] == 'HP':
                        cls_id = 0
                    else:
                        cls_id = 1   # SSA
                    self.datalist.append((img_path, cls_id))
                else:
                    continue

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img = Image.open(self.datalist[index][0]).convert('RGB')

        # label assignment
        label = int(self.datalist[index][1])

        #################
        # Convert PIL image to numpy array
        img = np.array(img)

        # First image
        img = self.transform1(image=img)
        img = Image.fromarray(img['image'])
        img = np.array(img)

        Aug1_img = self.transform4(image=img)
        Aug2_img = self.transform5(image=img)

        # Convert numpy array to PIL Image
        img = Image.fromarray(img)
        # img.show()
        Aug1_img = Image.fromarray(Aug1_img['image'])
        # Aug1_img.show()
        Aug2_img = Image.fromarray(Aug2_img['image'])
        # Aug2_img.show()

        # Convert to numpy array
        img = np.array(img)
        Aug1_img = np.array(Aug1_img)
        Aug2_img = np.array(Aug2_img)

        # Stack along specified dimension
        img = np.stack((img, Aug1_img, Aug2_img), axis=0)

        # Numpy to torch
        img = torch.from_numpy(img)

        # Randomize the augmentations
        shuffle_idx = torch.randperm(len(img))
        img = img[shuffle_idx, :, :, :]

        label = np.array(label)
        label = torch.from_numpy(label)
        label = label.repeat(img.shape[0])

        # Change Tensor Dimension to N x C x H x W
        img = img.permute(0, 3, 1, 2)

        return img, label


##### MHIST test loader
class DatasetMHIST_test:

    def __init__(self, dataset_path, annot_path, image_size):

        """
        TBLN dataset class wrapper (train with augmentation)
        """

        self.image_size = image_size

        # Resize images
        self.transform1 = Compose([Resize(image_size, image_size, interpolation=2)])

        # GT annotation
        GT = pd.read_csv(annot_path, header=None)

        self.datalist = []
        img_paths = glob.glob('{}/*.png'.format(dataset_path))
        with tqdm(enumerate(sorted(img_paths)), disable=True) as t:
            for wj, img_path in t:
                head, tail = os.path.split(img_path)
                img_id = tail  # Get image_id

                # check if it belongs to train/test set
                set = GT.loc[GT[0] == img_id][3]
                label = GT.loc[GT[0] == img_id][1]

                # Add only train/val to the corresponding set
                if set.iloc[0] == 'test':
                    if label.iloc[0] == 'HP':
                        cls_id = 0
                    else:
                        cls_id = 1   # SSA
                    self.datalist.append((img_path, cls_id))
                else:
                    continue

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img = Image.open(self.datalist[index][0]).convert('RGB')

        # label assignment
        label = int(self.datalist[index][1])

        #################
        # Convert PIL image to numpy array
        img = np.array(img)

        # First image
        img = self.transform1(image=img)
        img = Image.fromarray(img['image'])
        img = np.array(img)

        # Numpy to torch
        img = torch.from_numpy(img)
        label = np.array(label)
        label = torch.from_numpy(label)

        # Change Tensor Dimension to N x C x H x W
        img = img.permute(2, 0, 1)

        return img, label

#########################
