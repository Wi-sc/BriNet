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
from dataset2 import Dataset as dataset_val
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
print('Load model')
model = load_resnet_param(model, stop_layer='layer4', layer_num=50)
# load resnet-50 preatrained parameter
model = nn.DataParallel(model)
# disable the  gradients of not optomized layers
model_dic=torch.load(os.path.join(checkpoint_dir,modelname))
model.load_state_dict(model_dic, strict=False)
weight=model.state_dict()['module.conv_compress_23.0.weight'].clone()
bias=model.state_dict()['module.conv_compress_23.0.bias'].clone()
dic={}
dic['module.conv_compress_23.0.weight']=weight
dic['module.conv_compress_23.0.bias']=bias

def get_p(f):
    x=f.module.conv_compress_23.parameters()
    for y in x:
        yield y
    #yield f.module.conv_compress_23.parameters()
def get_f(f):
    yield f

optimizer=optim.SGD(get_p(model), lr=args.lr, weight_decay=0.0005)
all_inter, all_union,all_predict=[0]*5, [0]*5, [0]*5
each_inter, each_union=[], []
max_inter,max_union=[0]*5, [0]*5
zeros=[]
for k in range(args.online_iteration):
    tmp=[0]*20
    tmp2=[0]*20
    tmp3=[0]*20
    each_inter.append(tmp)
    each_union.append(tmp2)
    zeros.append(tmp3)
#turn_off(model)
model.cuda()
model = model.eval()
savedpic=[]
for i in range(args.online_iteration):
    savedpic.append([])
    for j in range(5):
        savedpic[i].append([])
size1=[161, 184, 209, 233, 257, 281, 305, 329]
size2=[545, 521, 497, 473, 449, 425, 401, 377]
sizelist=size1+size2
size1=size2
selflist=[]
savedir='writepic/'

def savepic(idx,name,data,iou):
    return
    data=str(data[0])
    fn=savedir+str(args.fold)+'-'+str(idx)+'-'+name+'-'+str(iou)+'.jpg'
    data=cv2.imread(os.path.join(args.data_dir,'JPEGImages',data+'.jpg')) 
    data=cv2.resize(data,(353,353))
    cv2.imwrite(fn,data)

def savemask(idx,name,data,iou):
    return
    fn=savedir+str(args.fold)+'-'+str(idx)+'-'+name+'-'+str(iou)+'.jpg'
    data=data.view(1,353,353)
    data=np.array(data)*255
    data=cv2.merge(data)
    data=np.expand_dims(data,axis=2)
    cv2.imwrite(fn,data)

for xxx in range(1):
    pom=[305,473]
    valset = dataset_val(data_dir=args.data_dir, mask_dir=args.mask_dir, fold=args.fold, input_size=args.input_size)
    valloader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    for i_iter, batch_val in enumerate(valloader):
        model.load_state_dict(model_dic,strict=False)
        #optimizer=optim.SGD(model.parameters(),lr=args.lr,momentum=0.9)
        query_img0,query_img1,query_img2,query_mask, sup_img, sup_mask, sample_class,query_name,support_name = batch_val
        savepic(i_iter,'support',support_name,1)
        savepic(i_iter,'query',query_name,1)
        query_img0=query_img0.cuda()
        query_img1=query_img1.cuda()
        query_img2=query_img2.cuda()
        sup_img=sup_img.cuda()
        savemask(i_iter,'st',sup_mask,1)
        sup_mask=sup_mask.cuda()
        
        savemask(i_iter,'gt',query_mask,1)
        query_mask=query_mask.cuda()
        pred=0
        smask=sup_mask[:,0,:,:]
        _,__,qh,qw=query_mask.shape
        _,__,sh,sw=sup_mask.shape
        pred_list=[]
        for k in range(args.online_iteration):
            continueflag=True
            with torch.no_grad():
                self_mask=model(sup_img,sup_img,sup_mask)
                self_mask=nn.functional.interpolate(self_mask,size=(sh,sw),mode='bilinear',align_corners=True)
                _,pred=torch.max(self_mask,dim=1)
                pred=pred.data.cpu()
                inter_list, union_list, _, num_predict_list = get_iou(sup_mask.cpu().long(), pred)
                self_iou=inter_list[0]/union_list[0]
                thr=(k+1)/args.online_iteration
                thr=1-(1+k)/(k+args.online_iteration)
                thr=thr*0.8
                if self_iou>-1+thr:
                    continueflag=False
            okflag=False
            for trys in range(3):
                if okflag:
                    break
                try:
                    attempt_mask=model(query_img1,sup_img,sup_mask)
                    mask_0=model(query_img0,sup_img,sup_mask)
                    mask_2=model(query_img2,sup_img,sup_mask)
                    okflag=True
                except:
                    print('forward re1',i_iter,k)
                    continue
            if not okflag:
                print('forward fail1',i_iter,k)
                continue
            attempt_mask=nn.functional.interpolate(attempt_mask, size=(qh,qw), mode='bilinear', align_corners=True)
            mask_1=attempt_mask
            mask_1=nn.functional.softmax(attempt_mask,dim=1)
            mask_0=nn.functional.interpolate(mask_0,size=(qh,qw),mode='bilinear',align_corners=True)
            mask_0=nn.functional.softmax(mask_0,dim=1)
            mask_2=nn.functional.interpolate(mask_2,size=(qh,qw),mode='bilinear',align_corners=True)
            mask_2=nn.functional.softmax(mask_2,dim=1)
            final_mask=mask_1+mask_0+mask_2
            attempt_mask=final_mask/3
            final_mask=nn.functional.softmax(final_mask,dim=1)
            #pic=attempt_mask.detach().cpu()
            #pic=np.asarray(pic).astype(np.float32)
            #savedpic[k][sample_class[0]-args.fold*5].append((query_name,support_name,pic))
            attempt_mask=final_mask[:,1:2,:,:]
            #attempt_mask*=(attempt_mask>0.5).float()
            #pred=pred.data.cpu()
            okflag=False
            for trys in range(3):
                if okflag:
                    break
                try:
                    res_mask=model(sup_img,query_img1,attempt_mask)
                    okflag=True
                except:
                    print('forward re2',i_iter,k)
                    continue
            if not okflag:
                print('forward fail2',i_iter,k)
                continue
            res_mask=nn.functional.interpolate(res_mask, size=(sh,sw), mode='bilinear', align_corners=True)
            loss=cross_entropy_calc_all(res_mask, smask)
            okflag=False
            for trys in range(3):
                if okflag:
                    break
                try:
                #if True:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    okflag=True
                except:
                    print('backward re',i_iter,k)
                    continue
            if not okflag:
                print('backward fail',i_iter,k)
                continue
        #    pred=iou*final_mask
            pred=final_mask
            _,pred=torch.max(pred,dim=1)
            pred=pred.data.cpu()
            inter_list, union_list, _, num_predict_list = get_iou(query_mask.cpu().long(), pred)
            if k==0:
                savemask(i_iter,'offline',pred,inter_list[0]/union_list[0])
            if continueflag:
                continue
            else:
                break
        for j in range(query_mask.shape[0]):
            each_inter[0][sample_class[j]-(args.fold*5)]+=inter_list[j]
            each_union[0][sample_class[j]-(args.fold*5)]+=union_list[j]
        savemask(i_iter,'online',pred,inter_list[0]/union_list[0])
    mx=[0]*20
    cnt1=0
    cnt2=0
    bestiter=[0]*20
    for k in range(1):
        iou=[0]*20
        cnt4=0
        for j in range(5):
            iou[j]=each_inter[k][j]/(1+each_union[k][j])
    #        print(j,k,iou[j])
            cnt4+=iou[j]
            if k==0:
                cnt1+=iou[j]
            if iou[j]>mx[j]:
                mx[j]=iou[j]
                bestiter[j]=k
        print(args.fold,k,cnt4/5)
