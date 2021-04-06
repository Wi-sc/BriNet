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
    def __init__(self, data_dir, mask_dir, fold, qinput_size=[321, 321], sinput_size=[321,321],normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.fold = fold
        self.img_list = self.get_img_list(fold=self.fold)
        self.pair_dict = self.get_pair()
        self.qinput_size = qinput_size
        self.sinput_size = sinput_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.initialized = False

    def get_img_list(self, fold):
        img_list = []
        fold_list = [0, 1, 2, 3]
        fold_list.remove(fold)
        for fold in fold_list:
            f = open(os.path.join(self.mask_dir, 'train', 'split%1d_train.txt' %fold))
            while True:
                item = f.readline()
                if item == '':
                    break
                img_name = item[:11]
                cat = int(item[13:15])
                img_list.append([img_name, cat])
        return img_list

    def get_pair(self):  # a list store all img name that contain that class
        pair_dict = {}
        for Class in range(1, 21):
            pair_dict[Class] = self.read_txt(os.path.join(self.mask_dir, 'train', '%d.txt' % Class))
        return pair_dict

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        return out_list

    def __getitem__(self, index):
        query_name = self.img_list[index][0]
        sample_class = self.img_list[index][1]  # random sample a class in this img
        support_img_list = self.pair_dict[sample_class]  # all img that contain the sample_class
        while True:  # random sample a support data
            support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
            if support_name != query_name:
                break
        support_mask = Image.open(os.path.join(self.mask_dir, 'train', str(sample_class), support_name + '.png'))
        support_img = Image.open(os.path.join(self.data_dir, 'JPEGImages', support_name + '.jpg'))
        query_mask = Image.open(os.path.join(self.mask_dir, 'train', str(sample_class), query_name + '.png'))
        query_img = Image.open(os.path.join(self.data_dir, 'JPEGImages', query_name + '.jpg'))
        support_img, support_mask = self.image_loader(self.sinput_size, support_img, support_mask)
        query_img, query_mask = self.image_loader(self.qinput_size, query_img, query_mask)
        if sample_class>(self.fold+1)*5:
            sample_class-=5
            
        return query_img, query_mask, support_img, support_mask, sample_class-1

    def image_loader(self, input_size, image, mask):
        # Resize
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
            
        scale_size = int(input_size[0] * random.uniform(1.0, 1.5))
        scale_size2=int(input_size[1] * random.uniform(1.0,1.5))
        scale_size=(scale_size, scale_size2)
        resize_img = transforms.Resize(size=scale_size, interpolation=Image.BILINEAR)
        image = resize_img(image)
        resize_mask = transforms.Resize(size=scale_size, interpolation=Image.NEAREST)
        mask = resize_mask(mask)
#         scale_size = int(input_size[0])
#         resize = transforms.Resize(size=(scale_size, scale_size), interpolation=Image.BILINEAR)
#         image = resize(image)
#         resize = transforms.Resize(size=(scale_size, scale_size), interpolation=Image.NEAREST)
#         mask = resize(mask)

        # Random rotation
        

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(input_size[0], input_size[1]))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
#         resize = transforms.Resize(size=(scale_size, scale_size), interpolation=Image.BILINEAR)
#         image = resize(image)
#         resize = transforms.Resize(size=(scale_size, scale_size), interpolation=Image.NEAREST)
#         mask = resize(mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.normalize_mean, self.normalize_std)
        mask = TF.to_tensor(mask)
        return image, mask

    def __len__(self):
        return len(self.img_list)
