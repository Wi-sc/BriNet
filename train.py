from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *
import time
import torch.nn.functional as F
import tqdm
import random
import argparse
from Dataset_train import Dataset as dataset_train
from Dataset_val import Dataset as dataset_val
import os
import torch
from Network import network
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
from args import args
import pandas as pd

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
                    help='SGD weight decay',
                    default=5e-4)
parser.add_argument('-momentum',
                    type=float,
                    help='SGD momentum',
                    default=0.9)
parser.add_argument('-fold',
                    type=int,
                    help='fold',
                    default=0)
parser.add_argument('-gpu',
                    type=str,
                    help='gpu id to use',
                    default='0')
parser.add_argument('-train_batch_size',
                    type=int,
                    help='train batch size',
                    default='16')
parser.add_argument('-val batch size',
                    type=int,
                    help='val batch size',
                    default='16')
parser.add_argument('-num_epoch',
                    type=int,
                    help='num epoch',
                    default=500)
parser.add_argument('-mask_dir',
                    type=str,
                    help='mask dir',
                    default='./data/Binary_map_aug')
parser.add_argument('-data_dir',
                    type=str,
                    help='data dir',
                    default='./data/VOC2012')
parser.add_argument('-checkpoint_dir',
                    type=str,
                    help='checkpoint dir',
                    default='./checkpoint/')
parser.add_argument('-alpha',
                    type=float,
                    help='aux seg loss weight',
                    default=1.0)

category = [['aeroplane', 'bicycle', 'bird', 'boat', 'bottle'],
                     ['bus', 'car', 'cat', 'chair', 'cow'],
                     ['diningtable', 'dog', 'horse', 'motorbike', 'person'],
                     ['potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
                        ]

args = parser.parse_args()
args.checkpoint_dir = os.path.join(args.checkpoint_dir, 'fold_%d'%args.fold)
checkpoint_dir = os.path.join(args.checkpoint_dir, 'fold_%d' %args.fold)
print(category[args.fold])

#set gpus
# gpu_list = [int(x) for x in args.gpu.split(',')]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

torch.backends.cudnn.benchmark = True
cudnn.enabled = True

# Create network.
model = network()

#load resnet50 preatrained parameter
model = load_resnet_param(model, stop_layer='fc', layer_num=50)
model=nn.DataParallel(model,[0,1])

# disable the  gradients of not optomized layers
turn_off(model)

if not os.path.exists(checkpoint_dir):
    os.makedirs(os.path.join(checkpoint_dir))


trainset = dataset_train(data_dir=args.data_dir, fold=args.fold, input_size=args.input_size)
trainloader = data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)

valset = dataset_val(data_dir=args.data_dir, fold=args.fold, input_size=args.input_size)
valloader = data.DataLoader(valset, batch_size=args.val_batch_size, shuffle=False, num_workers=4, drop_last=False)

save_pred_every =len(trainloader)
print('fold: %d Train: %d Val: %d'%(args.fold, len(trainset), len(valset)))

optimizer = optim.SGD([{'params': get_10x_lr_params(model), 'lr': args.lr}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


best_iou = 0.45
best_epoch = 0
history_doc = {'train loss':[], 
           'train final loss':[], 
           'train auxiliary loss':[], 
           'val loss':[], 
           'val iou':[], 
           category[args.fold][0], 
           category[args.fold][1], 
           category[args.fold][2], 
           category[args.fold][3], 
           category[args.fold][4]
            }
# history_doc = pd.read_csv(os.path.join(checkpoint_dir, 'train_log.csv'))

model.cuda()
model = model.train()
writer = SummaryWriter('./runs/Support2Conv')

for epoch in range(0, args.num_epoch):
    accumulated_loss = 0
    accumulated_final_loss = 0
    accumulated_aux_loss = 0
    begin_time = time.time()
#     tqdm_gen = tqdm.tqdm(trainloader)
    for i_iter, batch in enumerate(trainloader):
        query_img, query_mask, support_img, support_mask, sample_class, index = batch
        query_img = query_img.cuda()
        support_img = support_img.cuda()
        support_mask = support_mask.cuda()
        query_mask = query_mask.cuda()
        query_mask = query_mask[:, 0, :, :]
        optimizer.zero_grad()
        
        final_mask, aux_mask = model(query_img, support_img, support_mask)
        
        final_mask = nn.functional.interpolate(final_mask, size=args.input_size, mode='bilinear',align_corners=True)
        aux_mask = nn.functional.interpolate(aux_mask, size=args.input_size, mode='bilinear', align_corners=True)

        cross_entropy_final = cross_entropy_calc(final_mask, query_mask)
        cross_entropy_aux = cross_entropy_calc(aux_mask, query_mask)
        loss = cross_entropy_final + args.alpha*cross_entropy_aux
        loss.backward()
        optimizer.step()

        print('Epoch %3d: %4d/%d Loss = final + %.1f*aux: %.4f = %.4f + %.1f*%.4f'%(epoch, i_iter+1, save_pred_every, args.alpha, loss.item(), cross_entropy_final.item(), args.alpha, cross_entropy_aux.item()))
        # print('Epoch %3d: %4d/%d Loss = %.6f'%(epoch, i_iter+1, save_pred_every, loss.item()))

        #save training loss
        accumulated_loss += loss.item()
        accumulated_final_loss += cross_entropy_final.item()
        accumulated_aux_loss += cross_entropy_aux.item()
#         if i_iter % save_pred_every == 0 and i_iter != 0:
    history_doc['train loss'].append(accumulated_loss / save_pred_every)
    history_doc['train final loss'].append(accumulated_final_loss / save_pred_every)
    history_doc['train auxiliary loss'].append(accumulated_aux_loss / save_pred_every)
        
    print ('----Evaluation----')
    with torch.no_grad():
        accumulated_loss = 0
        model = model.eval()
        all_inter, all_union, all_predict = [0] * 5, [0] * 5, [0] * 5
        for i_iter, batch_ in enumerate(valloader):

            query_img, query_mask, support_img, support_mask, sample_class, index = batch_
            
            query_img = query_img.cuda()
            support_img = support_img.cuda()
            support_mask = support_mask.cuda()
            query_mask = query_mask.cuda()  # change formation for crossentropy use
            query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use

            pred = model(query_img, support_img, support_mask)
            pred = nn.functional.interpolate(pred, size=args.input_size, mode='bilinear', align_corners=True)  #upsample
            val_loss = cross_entropy_calc(pred, query_mask)
            accumulated_loss += val_loss.item()
            _, pred_label = torch.max(pred, 1)
            pred_label = pred_label.data.cpu()
            inter_list, union_list, _, num_predict_list = get_iou(query_mask.cpu().long(), pred_label)
            
            for j in range(query_mask.shape[0]):# watch out last drop last
                all_inter[sample_class[j] - (args.fold * 5 + 1)] += inter_list[j]
                all_union[sample_class[j] - (args.fold * 5 + 1)] += union_list[j]
        history_doc['val loss'].append(accumulated_loss/len(valloader))
        
        IOU = [0] * 5
        for j in range(5):
            IOU[j] = all_inter[j] / all_union[j]
            print('Category:', category[args.fold][j], IOU[j])
            history_doc[category[args.fold][j]].append(IOU[j])
                
        
        mean_iou = np.mean(IOU)
        
        history_doc['val iou'].append(mean_iou)
        print('Epoch: %d | IOU: %.4f | Learning rate: %.7f' % (epoch, mean_iou, optimizer.param_groups[0]['lr']))

        if mean_iou > best_iou:
            best_iou = mean_iou
            model = model.eval()
            torch.save(model.cpu().state_dict(), os.path.join(checkpoint_dir, 'model-%d-%.4f-%.4f.pth'%(epoch, history_doc['val loss'][-1], mean_iou)))
            model = model.train()
            best_epoch = epoch
            print('A better model is saved')

#         print('Best IOU Up to Now: %.4f' % (best_iou))
        
        
        model = model.train()
        model.cuda()
    
    
    epoch_time = time.time() - begin_time
    pd.DataFrame(history_doc).to_csv(os.path.join(checkpoint_dir, 'train_log.csv'), index=False)
    print('Best epoch:%d ,iout:%.4f' % (best_epoch, best_iou))
    print('This epoch takes:', epoch_time/3600, 'hours')
    print('Still need %.4f hours' % ((args.num_epoch - epoch) * epoch_time / 3600))



