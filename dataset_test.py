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
    def __init__(self, data_dir, name_idx, fold, input_size=[224, 224], normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        self.data_dir = data_dir
        self.fold = fold
        self.pair_list = self.get_img_list(fold=self.fold)
        self.input_size = input_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.name_idx = name_idx

    def get_pair_list(self, fold):
        pair_list = []
        cls_list = []
        f = open(os.path.join(self.mask_dir, 'train', 'split%1d_train.txt' %fold))
        f = open('./data/Binary_map_aug/val/val_set.txt')
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
        support_mask = Image.open(os.path.join(self.data_dir, support_name+'.png')).convert('1')
        support_img = Image.open(os.path.join(self.data_dir, support_name+'.jpg')).convert("RGB")
        query_mask = Image.open(os.path.join(self.data_dir, query_name+'.png')).convert('1')
        query_img = Image.open(os.path.join(self.data_dir, query_name+'.jpg')).convert("RGB")
        _, support_img, _, support_mask = self.image_process(self.input_size, support_img, support_mask)
        query_img0, query_img1, query_img2, query_mask = self.image_process(self.input_size, query_img, query_mask)
        return query_img0, query_img1, query_img2, query_mask, support_img, support_mask, class_name

    def image_process(self, input_size, image, mask):
        h, w =input_size
        #h,w=image.size
        resize=transforms.Resize(size=(h, w),interpolation=Image.NEAREST)
        mask=resize(mask)
        resize=transforms.Resize(size=(h, w),interpolation=Image.BILINEAR)
        image0=resize(image)

        # mutil-scale evaluation ([305, 305], [353,353], [473, 473])
        resize=transforms.Resize(size=(305, 305),interpolation=Image.BILINEAR)
        image1=resize(image)
        
        resize=transforms.Resize(size=(473, 473),interpolation=Image.BILINEAR)
        image2=resize(image)
        
        image0 = TF.to_tensor(image0)
        image0 = TF.normalize(image0, self.normalize_mean, self.normalize_std)
        image1 = TF.to_tensor(image1)
        image1 = TF.normalize(image1, self.normalize_mean, self.normalize_std)
        image2 = TF.to_tensor(image2)
        image2 = TF.normalize(image2, self.normalize_mean, self.normalize_std)
        mask = TF.to_tensor(mask)
        return image0, image1, image2, mask

        

    def __len__(self):
        return len(self.pair_list)