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
    def __init__(self, data_dir, name_idx, input_size=[224, 224], normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        self.data_dir = data_dir
        self.pair_list = self.get_pair_list()
        self.input_size = input_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.name_idx=name_idx

    def get_pair_list(self):
        pair_list = []
        cls_list = []
        f = open('./val_set.txt')
        line = f.readline()
        while line:
#             filelist = os.listdir(os.path.join(self.data_dir, line[:-1]))
#             for sup_name in range(1, 11):
#                 for query_name in range(1, 11):
#                     if(sup_name!=query_name):
#                         pair_list.append([line[:-1], line[:-1]+'/'+str(sup_name), line[:-1]+'/'+str(query_name)])
            cls_list.append(line[:-1])
            line = f.readline()
        for sup_name in range(1, 11):
            for query_name in range(1, 11):
                if(sup_name!=query_name):
                    for cls in cls_list:
                        pair_list.append([cls, cls+'/'+str(sup_name), cls+'/'+str(query_name)])
        return pair_list

    def __getitem__(self, index):
        support_name = self.pair_list[index][1]
        query_name = self.pair_list[index][2]
        class_name = self.pair_list[index][0]  # random sample a class in this img
        support_mask = Image.open(os.path.join(self.data_dir, support_name+'.png')).convert('1')
        support_img = Image.open(os.path.join(self.data_dir, support_name+'.jpg')).convert("RGB")
        query_mask = Image.open(os.path.join(self.data_dir, query_name+'.png')).convert('1')
        query_img = Image.open(os.path.join(self.data_dir, query_name+'.jpg')).convert("RGB")
        support_img, support_mask = self.image_process(self.input_size, support_img, support_mask)
        query_img, query_mask = self.image_process(self.input_size, query_img, query_mask)
        return query_img, query_mask, support_img, support_mask, self.name_idx[class_name]

    def image_process(self, input_size, image, mask):
        # Resize rand crop
#         scale_rate = random.uniform(1,1.5)
#         scale_size = int(input_size[0]*scale_rate)
#         resize = transforms.Resize(size=(scale_size, scale_size), interpolation=Image.BILINEAR)
#         image = resize(image)
#         resize = transforms.Resize(size=(scale_size, scale_size), interpolation=Image.NEAREST)
#         mask = resize(mask)

        # Resize
#         scale_size = input_size
        
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