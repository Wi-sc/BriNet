from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *
import time
import torch.nn.functional as F
import random
import argparse
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from dataset_val import Dataset as dataset_val
from BriNet import network
parser = argparse.ArgumentParser()

parser.add_argument('-lr',
                    type=float,
                    help='SGD learning rate',
                    default=0.025)
parser.add_argument('-input_size',
                    type=list,
                    help='input_size',
                    default=[353, 353])
parser.add_argument('-weight_decay',
                    type=float,
                    help='SGD weight decay'
                    ,
                    default=5e-4)
parser.add_argument('-fold',
                    type=int,
                    help='fold',
                    default=0)
parser.add_argument('-gpu',
                    type=str,
                    help='gpu id to use',
                    default='0')
parser.add_argument('-batch_size',
                    type=int,
                    help='batch size',
                    default='32')
parser.add_argument('-num_epoch',
                    type=int,
                    help='num epoch',
                    default=1000)
parser.add_argument('-mask_dir',
                    type=str,
                    help='mask dir',
                    default='./data/Binary_map_aug/')
parser.add_argument('-data_dir',
                    type=str,
                    help='data dir',
                    default='./data/VOC2012')
parser.add_argument('-checkpoint_dir',
                    type=str,
                    help='checkpoint dir',
                    default='./checkpoint/')
parser.add_argument('-online_iteration', 
                    type=int, 
                    default=20)

args = parser.parse_args()
print(args)
category = [['aeroplane', 'bicycle', 'bird', 'boat', 'bottle'],
            ['bus', 'car', 'cat', 'chair', 'cow'],
            ['diningtable', 'dog', 'horse', 'motorbike', 'person'],
            ['potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
            ]
print(category[args.fold])
checkpoint_dir = os.path.join(args.checkpoint_dir, 'fold_%d' % args.fold)

bottom_line=0
for file in os.listdir(checkpoint_dir):
    if file[:5]=='model' and len(file)>20:
        model_acc = float(file.split('-')[3][:6])
        if model_acc>bottom_line:
            modelname=file
            bottom_line=model_acc
print(modelname)
torch.backends.cudnn.benchmark = True

cudnn.enabled = True

# Create network.

model = network()
# load resnet-50 preatrained parameter
model = load_resnet_param(model, stop_layer='layer4', layer_num=50)
model = nn.DataParallel(model)
# disable the  gradients of not optomized layers
model_dic=torch.load(os.path.join(checkpoint_dir,modelname))
model.load_state_dict(model_dic, strict=False)
weight=model.state_dict()['module.channel_compress.0.weight'].clone()
bias=model.state_dict()['module.channel_compress.0.bias'].clone()

def get_parameters(f):
    x=f.module.channel_compress.parameters()
    for y in x:
        yield y

optimizer=optim.SGD(get_parameters(model), lr=args.lr, weight_decay=0.0005)
all_inter, all_union,all_predict=[0]*5, [0]*5, [0]*5

model.cuda()
model = model.eval()

valset = dataset_val(data_dir=args.data_dir, mask_dir=args.mask_dir, fold=args.fold, input_size=args.input_size)
valloader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
for i_iter, batch_val in enumerate(valloader):
    model.load_state_dict(model_dic, strict=False)
    query_img0, query_img1, query_img2, query_mask, sup_img, sup_mask, sample_class = batch_val
    query_img0 = query_img0.cuda()
    query_img1 = query_img1.cuda()
    query_img2 = query_img2.cuda()
    sup_img = sup_img.cuda()
    sup_mask = sup_mask.cuda()
    query_mask = query_mask.cuda()
    pred = 0
    _, _, qh, qw = query_mask.shape
    _, _, sh, sw = sup_mask.shape
    pred_list = []
    for k in range(args.online_iteration):
        model = model.eval()
        with torch.no_grad():
            self_mask = model(sup_img, sup_img, sup_mask)
            self_mask = nn.functional.interpolate(self_mask, size=(sh,sw), mode='bilinear', align_corners=True)
            _, pred = torch.max(self_mask,dim=1)
            pred = pred.data.cpu()
            inter_list, union_list, _, num_predict_list = get_iou(sup_mask.cpu().long(), pred)
            self_iou=inter_list[0]/union_list[0]
            thr=(args.online_iteration-1)/(k+args.online_iteration)*0.8
            if self_iou>thr:
                break
        
        model = model.train()
        mask_0 = model(query_img0, sup_img, sup_mask)
        mask_1 = model(query_img1, sup_img, sup_mask)
        mask_2 = model(query_img2, sup_img, sup_mask)
        mask_0 = nn.functional.interpolate(mask_0, size=(qh,qw), mode='bilinear', align_corners=True)
        mask_0 = nn.functional.softmax(mask_0, dim=1)
        mask_1 = nn.functional.interpolate(mask_1, size=(qh,qw), mode='bilinear', align_corners=True)
        mask_1 = nn.functional.softmax(mask_1, dim=1)
        mask_2 = nn.functional.interpolate(mask_2, size=(qh,qw), mode='bilinear', align_corners=True)
        mask_2 = nn.functional.softmax(mask_2, dim=1)
        final_mask = mask_1 + mask_0 + mask_2
        final_mask = nn.functional.softmax(final_mask,dim=1)
        res_mask = model(sup_img, query_img1, final_mask[:,1:2,:,:])
        res_mask = nn.functional.interpolate(res_mask, size=(sh, sw), mode='bilinear', align_corners=True)
        loss=cross_entropy_calc_all(res_mask, sup_mask[:,0,:,:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model = model.eval()
        mask_0 = model(query_img0, sup_img, sup_mask)
        mask_1 = model(query_img1, sup_img, sup_mask)
        mask_2 = model(query_img2, sup_img, sup_mask)
        mask_0 = nn.functional.interpolate(mask_0, size=(qh,qw), mode='bilinear', align_corners=True)
        mask_0 = nn.functional.softmax(mask_0, dim=1)
        mask_1 = nn.functional.interpolate(mask_1, size=(qh,qw), mode='bilinear', align_corners=True)
        mask_1 = nn.functional.softmax(mask_1, dim=1)
        mask_2 = nn.functional.interpolate(mask_2, size=(qh,qw), mode='bilinear', align_corners=True)
        mask_2 = nn.functional.softmax(mask_2, dim=1)
        pred = mask_1 + mask_0 + mask_2
        pred = nn.functional.softmax(pred, dim=1)
        _, pred = torch.max(pred, dim=1)
        pred = pred.data.cpu()
        inter_list, union_list, _, num_predict_list = get_iou(query_mask.cpu().long(), pred)

    for j in range(query_mask.shape[0]):
        all_inter[sample_class[j]-(args.fold*5)]+=inter_list[j]
        all_union[sample_class[j]-(args.fold*5)]+=union_list[j]

iou=0
for j in range(5):
    iou+=all_inter[j]/all_union[j]
    print(category[args.fold][j], all_inter[j]/all_union[j])
print('fold ', args.fold, 'Mean IoU: ', iou/5)
