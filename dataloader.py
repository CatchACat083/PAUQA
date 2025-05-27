from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.functional import resize, rotate, crop, hflip, vflip, to_tensor, normalize

from PIL import Image
from cProfile import label
import os
import random
import re

import numpy as np
import pandas as pd
import torch 

from PIL import Image

def data_loader(dataset_name, batch_size):        
    if dataset_name == 'LUIQD':
        image_path = "./data/LUIQD/"
        train_label = "./data/LUIQD/db_train_final.csv"
        valid_label = "./data/LUIQD/db_valid_final.csv"
        
        train_dataset = Underwater_Set(file_path = image_path, label_path = train_label, status='train', augmentation=True,
                               angle=2, crop_size_h=384, crop_size_w=384, hflip_p=0.5)
        valid_dataset = Underwater_Set(file_path = image_path, label_path = valid_label, status='valid', augmentation=False,
                               angle=0, crop_size_h=384, crop_size_w=384, hflip_p=0)
        
    elif dataset_name == 'LUIQD_TEST':
        image_path = "./data/LUIQD/"
        train_label = "./data/LUIQD/db_train_final.csv"
        valid_label = "./data/LUIQD/db_valid_final.csv"
        
        train_dataset = Underwater_Set_withname(file_path = image_path, label_path = train_label, status='train', augmentation=True,
                               angle=2, crop_size_h=384, crop_size_w=384, hflip_p=0.5)
        valid_dataset = Underwater_Set_withname(file_path = image_path, label_path = valid_label, status='valid', augmentation=False,
                               angle=0, crop_size_h=384, crop_size_w=384, hflip_p=0)
    else:
        print("!!! Dataset" + dataset_name + "name not matched.")

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False, num_workers = 8)

    return train_loader, valid_loader

def RandomCropPatches(im, patch_size=32, n_patches=32):
    """
    Random Crop Patches
    :param im: the distorted image
    :param ref: the reference image if FR-IQA is considered (default: None)
    :param patch_size: patch size (default: 32)
    :param n_patches: numbers of patches (default: 32)
    :return: patches
    """
    w, h = im.size

    patches = ()
    for i in range(n_patches):
        w1 = np.random.randint(low=0, high=w-patch_size+1)
        h1 = np.random.randint(low=0, high=h-patch_size+1)
        patch = to_tensor(im.crop((w1, h1, w1 + patch_size, h1 + patch_size)))
        patches = patches + (patch,)

    return torch.stack(patches)

def NonOverlappingCropPatches(im, patch_size=32):
    """
    NonOverlapping Crop Patches
    :param im: the distorted image
    :param patch_size: patch size (default: 32)
    :return: patches
    """
    w, h = im.size

    patches = ()
    stride = patch_size
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patches = patches + (patch,)

    return torch.stack(patches)

class Underwater_Set(torch.utils.data.Dataset):
    def __init__(self, file_path, label_path, status, augmentation, angle, crop_size_h, crop_size_w, hflip_p):
        super(Underwater_Set).__init__()
        self.status = status
        self.augment = augmentation
        self.angle = angle
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.hflip_p = hflip_p

        self.__load_data__(file_path, label_path)

    def __getitem__(self, index):
        (image_name, image_score) = self.data[index]
        image = self.transform(Image.open(image_name).convert('RGB'),
                               self.status, self.angle, self.crop_size_h, self.crop_size_w, self.hflip_p)
        score = torch.tensor(image_score, dtype = torch.float32)
        return image, score
    
    def __len__(self):
        return len(self.data)

    def __load_data__(self, file_path, label_path):
        image_files = os.listdir(file_path)
        image_labels = pd.read_csv(label_path).values.tolist()

        self.data = []
        for i in range(len(image_labels)):
            self.data.append((os.path.join(file_path, image_labels[i][0]), image_labels[i][1]))
        print(">>> DATALOADER >>> " + self.status + " data size is :" + str(len(self.data)))

    def transform(self, im, status, angle=2, crop_size_h=224, crop_size_w=224, hflip_p=0.5):
        if status == 'train' and self.augment:  # data augmentation
            w, h = im.size

            im = resize(im, (self.crop_size_h, self.crop_size_w))
            if np.random.rand(1) < 0.5:  # flip horizonly
                im = hflip(im)
            # if np.random.rand(1) < 0.5:  ## flip vertically
            #     im = vflip(im)

        else:
            im = resize(im, (self.crop_size_h, self.crop_size_w))

        im = to_tensor(im)
        # im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        return im
    
class Underwater_Set_withname(torch.utils.data.Dataset):
    def __init__(self, file_path, label_path, status, augmentation, angle, crop_size_h, crop_size_w, hflip_p):
        super(Underwater_Set).__init__()
        self.status = status
        self.augment = augmentation
        self.angle = angle
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.hflip_p = hflip_p

        self.__load_data__(file_path, label_path)

    def __getitem__(self, index):
        (image_name, image_score) = self.data[index]
        image = self.transform(Image.open(image_name).convert('RGB'),
                               self.status, self.angle, self.crop_size_h, self.crop_size_w, self.hflip_p)
        score = torch.tensor(image_score, dtype = torch.float32)
        return image, score, image_name
    
    def __len__(self):
        return len(self.data)

    def __load_data__(self, file_path, label_path):
        image_files = os.listdir(file_path)
        image_labels = pd.read_csv(label_path).values.tolist()

        self.data = []
        for i in range(len(image_labels)):
            self.data.append((os.path.join(file_path, image_labels[i][0]), image_labels[i][1]))
        print(">>> DATALOADER >>> " + self.status + " data size is :" + str(len(self.data)))

    def transform(self, im, status, angle=2, crop_size_h=224, crop_size_w=224, hflip_p=0.5):
        if status == 'train' and self.augment:  # data augmentation
            w, h = im.size

            im = resize(im, (self.crop_size_h, self.crop_size_w))
            if np.random.rand(1) < 0.5:  # flip horizonly
                im = hflip(im)
            # if np.random.rand(1) < 0.5:  ## flip vertically
            #     im = vflip(im)

        else:
            im = resize(im, (self.crop_size_h, self.crop_size_w))

        im = to_tensor(im)
        # im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        return im