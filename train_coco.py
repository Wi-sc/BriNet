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
import numpy as np
from coco_train import Dataset as dataset_train
from coco_val2 import Dataset as dataset_val
#from backbone_afterMaskPooling import network
#from backbone_before import network
from backbone_nn import network
parser = argparse.ArgumentParser()

parser.add_argument('-lr',
                    type=float,
                    help='SGD learning rate',
                    default=0.0025)
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
                    default='/mnt/lustre/share/wangbairun/mscoco2014_mask/')
parser.add_argument('-data_dir',
                    type=str,
                    help='data dir',
                    default='/mnt/lustre/share/zhangmingyuan/datasets/mscoco2014')
parser.add_argument('-checkpoint_dir',
                    type=str,
                    help='checkpoint dir',
                    default='/mnt/lustre/wangbairun/CANet_Modified/channelwise_attention/checkpoint/')
parser.add_argument('-alpha',
                    type=float,
                    help='loss weight',
                    default=1.0)
parser.add_argument('-beta',
                    type=float,
                    help='aux seg loss weight',
                    default=1.0)

# config = Config()
args = parser.parse_args()
print(args)


category = [['aeroplane', 'bicycle', 'bird', 'boat', 'bottle'],
                     ['bus', 'car', 'cat', 'chair', 'cow'],
                     ['diningtable', 'dog', 'horse', 'motorbike', 'person'],
                     ['potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
                        ]
# checkpoint_dir = os.path.join(args.checkpoint_dir, 'fold_%d' % args.fold)
checkpoint_dir = args.checkpoint_dir
print(category[args.fold])
log_path = os.path.join(checkpoint_dir, 'train_log_w%e.csv' %args.weight_decay)
print('train log path: ', log_path)

# set gpus
# gpu_list = [int(x) for x in config.gpu.split(',')]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

torch.backends.cudnn.benchmark = True

cudnn.enabled = True

# Create network.

model = network()
# print('Load model with template')

model = nn.DataParallel(model)
# load resnet-50 preatrained parameter
model = load_resnet_param(model, stop_layer='layer4', layer_num=50)

# disable the  gradients of not optomized layers

turn_off(model, is_layer4_available=False)

if not os.path.exists(checkpoint_dir):
    os.makedirs(os.path.join(checkpoint_dir))
    
bottom_line_list = [0.2]*4
bottom_line = bottom_line_list[args.fold]
model_name = None
for file in os.listdir(checkpoint_dir):
    if file[:5]=='model' and len(file)>20:
        model_acc = float(file.split("-")[3][:6])
        if model_acc>=bottom_line:
            model_name = file
            bottom_line = model_acc
if model_name != None:
    print("Load model path: ", os.path.join(checkpoint_dir, model_name))
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, model_name)),strict=False)

dataset = dataset_train(data_dir=args.data_dir, mask_dir=args.mask_dir, fold=args.fold, input_size=args.input_size)
trainloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

valset = dataset_val(data_dir=args.data_dir, mask_dir=args.mask_dir, fold=args.fold, input_size=args.input_size)
valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

save_pred_every = len(trainloader)
print('fold: %d Train: %d Val: %d' % (args.fold, len(dataset), len(valset)))

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


best_iou = 0
best_epoch = -1
history_doc = ['train loss',
               'val loss',
               category[args.fold][0],
               category[args.fold][1],
               category[args.fold][2],
               category[args.fold][3],
               category[args.fold][4],
                'val iou'
               ]
# wirte_csv(history_doc, log_path)


model.cuda()
model = model.train()

for epoch in range(args.num_epoch):
#     if epoch>=args.num_epoch-10:
#         turn_on(model, is_layer4_available=False)
    
    adjust_learning_rate(optimizer, args.lr, epoch, 20)
    accumulated_loss = 0
    begin_time = time.time()
    history_doc = []
    for i_iter, batch_data in enumerate(trainloader):
        query_img, query_mask, support_img, support_mask, sample_class, index = batch_data
        query_img = query_img.cuda()
        support_img = support_img.cuda()
        support_mask = support_mask.cuda()
        query_mask = query_mask.cuda()  # change formation for crossentropy use
        # remove the second dim,change formation for crossentropy use
        sample_class = sample_class.cuda()
        optimizer.zero_grad()
            
        final_mask, pre_mask,cls_s = model(query_img, support_img, support_mask)
#         final_mask, cls_s = model(query_img, support_img, support_mask)

        final_mask = nn.functional.interpolate(final_mask, size=args.input_size, mode='bilinear', align_corners=True)
        pre_mask = nn.functional.interpolate(pre_mask, size=args.input_size, mode='bilinear', align_corners=True)
        query_mask = query_mask[:, 0, :, :]

#         cross_entropy_final = cross_entropy_calc(final_mask, query_mask, weight=torch.tensor([1., 5.]))
#         cross_entropy_aux = cross_entropy_calc(pre_mask, query_mask, weight=torch.tensor([1., 5.]))
        cross_entropy_final = cross_entropy_calc(final_mask, query_mask)
        cross_entropy_aux = cross_entropy_calc(pre_mask, query_mask)
        cross_entropy_s = cross_entropy_calc(cls_s,sample_class)
#         cross_entropy_q = cross_entropy_calc(cls_q, sample_class)
#         loss = cross_entropy_final + args.alpha*cross_entropy_s
        loss = cross_entropy_final + args.beta*cross_entropy_aux + args.alpha*cross_entropy_s 
#         loss = cross_entropy_final
        #loss = cross_entropy_aux + args.alpha*cross_entropy_s
        loss.backward()
        optimizer.step()
        
        _, pred_cls = torch.max(cls_s, dim=1)
        acc = (pred_cls == sample_class).sum().item()/sample_class.size(0)
        # 2-loss
        if i_iter%40==0:
            print('Epoch %3d: %4d/%d | Loss: %.6f = %.6f + %.1f*%.6f + %.1f*%.6f | Accuracy: %.4f' 
              % (epoch, i_iter+1, save_pred_every, loss.item(), cross_entropy_final.item(), args.beta, cross_entropy_aux.item(), args.alpha, cross_entropy_s, acc))
#             print('Epoch %3d: %4d/%d | Loss: %.6f = %.6f + %.1f*%.6f | Accuracy: %.4f' 
#               % (epoch, i_iter+1, save_pred_every, loss.item(), cross_entropy_final.item(), args.alpha, cross_entropy_s.item(), acc))
        
        accumulated_loss += loss.item()
    history_doc.append(accumulated_loss / save_pred_every)

    print('----Evaluation----')
    with torch.no_grad():
        accumulated_loss = 0
        model = model.eval()
        all_inter, all_union, all_predict = [0] * 20, [0] * 20, [0] * 20
        for i_iter, batch_data in enumerate(valloader):
            query_img, query_mask, support_img, support_mask, sample_class = batch_data
            query_img = query_img.cuda()
            query_mask = query_mask.cuda()
            support_img = support_img.cuda()
            support_mask = support_mask.cuda()
            query_mask = query_mask[:, 0, :, :]
            final_mask = model(query_img, support_img, support_mask)
            final_mask = nn.functional.interpolate(final_mask, size=query_mask.shape[-2:], mode='bilinear', align_corners=True)
            cross_entropy_final = cross_entropy_calc(final_mask, query_mask)
            accumulated_loss += cross_entropy_final.item()
            _, pred_label = torch.max(final_mask, 1)
            pred_label = pred_label.data.cpu() 
            inter_list, union_list, _, num_predict_list = get_iou(query_mask.cpu().long(), pred_label)

            for j in range(query_mask.shape[0]):  # watch out last drop last!!(!=batch size)
                all_inter[sample_class[j] - (args.fold * 0)] += inter_list[j]
                all_union[sample_class[j] - (args.fold * 0)] += union_list[j]
            
        history_doc.append(accumulated_loss / len(valloader))

        IOU = [0] * 20
        for j in range(20):
            IOU[j] = all_inter[j] / all_union[j]
            history_doc.append(IOU[j])

        mean_iou = np.mean(IOU)

        history_doc.append(mean_iou) # val iou
        print('Epoch: %d | IOU: %.4f | Learning rate: %.7f' % (epoch, mean_iou, optimizer.param_groups[0]['lr']))

        if mean_iou > best_iou:
            if mean_iou>best_iou:
                best_iou = mean_iou
                best_epoch = epoch
            if mean_iou > bottom_line:
                model = model.eval()
                torch.save(model.cpu().state_dict(), os.path.join(checkpoint_dir,'model-%d-%.4f-%.4f-%e.pth' %(epoch, history_doc[1], mean_iou, args.weight_decay)))
                model = model.train()
                print('A better model is saved')
            

        #         print('Best IOU Up to Now: %.4f' % (best_iou))

        model = model.train()
        model.cuda()

    
    epoch_time = time.time() - begin_time
#     wirte_csv(history_doc, log_path)
    print('Best epoch: %d, Best IoU: %.4f' % (best_epoch, best_iou))
    print('This epoch takes:', epoch_time / 60, 'mins')
    print('Still need %.4f hours' % ((args.num_epoch - epoch) * epoch_time / 3600))



