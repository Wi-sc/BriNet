import random
import os
import io
import torchvision
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F_tensor
import numpy as np
from torch.utils.data import DataLoader

class Dataset(object):
    def __init__(self, data_dir, mask_dir, fold, input_size=[224, 224], normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.fold=fold
        self.pair_list = self.get_pair_list()
        self.input_size = input_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
    def get_pair_list(self):
        pair_list = []
        cls_list = []
        f = open('./data/Binary_map_aug/val/split%1d_val.txt' %self.fold)
        line = f.readline()
        while line:
            sup_name, query_name, cat = line.split()
            cat = int(cat)
            pair_list.append([query_name, sup_name, cat])
            line = f.readline()
        return pair_list

    def __getitem__(self, index):
        query_name = self.pair_list[index][0]
        support_name = self.pair_list[index][1]
        class_name = self.pair_list[index][2]  # random sample a class in this img
        support_mask = Image.open(os.path.join(self.mask_dir, support_name+'.png')).convert('1')
        support_img = Image.open(os.path.join(self.data_dir, support_name+'.jpg')).convert("RGB")
        query_mask = Image.open(os.path.join(self.mask_dir, query_name+'.png')).convert('1')
        query_img = Image.open(os.path.join(self.data_dir, query_name+'.jpg')).convert("RGB")
        support_img, support_mask = self.image_process(self.input_size, support_img, support_mask)
        query_img, query_mask = self.image_process(self.input_size, query_img, query_mask)
        return query_img, query_mask, support_img, support_mask, class_name

    def image_process(self, input_size, image, mask):
        assert mask.size == image.size
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.normalize_mean, self.normalize_std)
        mask = np.asarray(mask)
        mask = np.where(mask>0, 1, 0)
        mask = np.float32(mask)
        mask = TF.to_tensor(mask)
        return image, mask

        

    def __len__(self):
        return len(self.pair_list)
