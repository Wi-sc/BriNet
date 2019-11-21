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
from Config import Config
import pandas as pd



parser = argparse.ArgumentParser()


parser.add_argument('-lr',
                    type=float,
                    help='learning rate',
                    default=0.0025)


parser.add_argument('-fold',
                    type=int,
                    help='fold',
                    default=0)

parser.add_argument('-gpu',
                    type=str,
                    help='gpu id to use',
                    default='0,1')
parser.add_argument('-batch_size',
                    type=int,
                    help='batch_size',
                    default='16')


config = Config()
args = parser.parse_args()
config.lr = args.lr
# config.lr = config.lr/10
config.num_epoch = 200
config.train_batch_size = args.batch_size
config.val_batch_size = args.batch_size
config.fold = args.fold
config.gpu = args.gpu
# config.checkpoint_dir = os.path.join(config.checkpoint_dir, fold_%d'config.fold )
# config.checkpoint_dir = './checkpoint/fold_%d'%config.fold
checkpoint_dir = os.path.join(config.checkpoint_dir, 'fold_%d' %config.fold)
print(config.category[config.fold])

#set gpus
# gpu_list = [int(x) for x in config.gpu.split(',')]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

torch.backends.cudnn.benchmark = True






cudnn.enabled = True


# Create network.
model = network()
#load resnet-50 preatrained parameter
model = load_resnet_param(model, stop_layer='fc', layer_num=50)
model=nn.DataParallel(model,[0,1])

# disable the  gradients of not optomized layers
# turn_off(model)


if not os.path.exists(checkpoint_dir):
    os.makedirs(os.path.join(checkpoint_dir))
# model-56-0.2222-0.5108.pth
# model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'model-190-0.2101-0.5112.pth')))



dataset = dataset_train(data_dir=config.data_dir, fold=config.fold, input_size=config.input_size, normalize_mean=config.IMG_MEAN,
                  normalize_std = config.IMG_STD)
trainloader = data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4)

valset = dataset_val(data_dir=config.data_dir, fold=config.fold, input_size=config.input_size, normalize_mean=config.IMG_MEAN,
                  normalize_std = config.IMG_STD)
valloader = data.DataLoader(valset, batch_size=config.val_batch_size, shuffle=False, num_workers=4, drop_last=False)

save_pred_every =len(trainloader)
print('fold: %d Train: %d Val: %d'%(config.fold, len(dataset), len(valset)))

optimizer = optim.SGD([{'params': get_10x_lr_params(model), 'lr': 10 * config.lr}],
                          lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
# opt_Adam = optim.Adam(get_adam_params(model), lr = 0.00001, betas=(0.9, 0.99))


best_iou = 0.5
best_epoch = 0
# history_doc = {'train loss':[], 'train cos similarity loss':[], 'train cross entropy loss':[],
#            'val loss':[], 'val cos similarity loss':[], 'val cross entropy loss':[], 'val iou':[], }
history_doc = {'train loss':[], 
#            'train mse loss':[], 
#            'train cross entropy loss':[], 
           'val loss':[], 
           'val iou':[], 
           config.category[config.fold][0]:[], 
           config.category[config.fold][1]:[], 
           config.category[config.fold][2]:[], 
           config.category[config.fold][3]:[], 
           config.category[config.fold][4]:[]
            }
# history_doc = pd.read_csv(os.path.join(checkpoint_dir, 'train_log.csv'))
# history_doc = history_doc[:191].to_dict('list')

model.cuda()
model = model.train()
writer = SummaryWriter('./runs/Support2Conv')

for epoch in range(0, config.num_epoch):
    accumulated_loss = 0
#     accumulated_cross_entropy = 0
#     accumulated_mse = 0
    begin_time = time.time()
#     tqdm_gen = tqdm.tqdm(trainloader)
    for i_iter, batch in enumerate(trainloader):
        query_img, query_mask, support_img, support_mask, sample_class, index = batch
        query_img = query_img.cuda()
        support_img = support_img.cuda()
        support_mask = support_mask.cuda()
        query_mask = query_mask.cuda()  # change formation for crossentropy use
          # remove the second dim,change formation for crossentropy use
        optimizer.zero_grad()
        
        pred = model(query_img, support_img, support_mask)
        
#         pred = nn.functional.interpolate(torch.unsqueeze(pred, 1), size=config.input_size, mode='bilinear',align_corners=True) # upsample 
        pred = nn.functional.interpolate(pred, size=config.input_size, mode='bilinear', align_corners=True)
#         pred = torch.squeeze(pred, 1)
        
#         print(torch.max(support_mask[0]))
        
#         support_masked_features = get_masked_features(support_feature_maps, support_mask)
#         query_masked_features = get_masked_features(query_feature_maps, query_mask)
#         print(support_features.type(), support_masked_features.type())
#         print(query_features.type(), query_masked_features.type())
#         cos_similarity = cos_similarity_calc(inter_feature_maps, query_masked_feature_maps)
#         mse = mse_calc(inter_features, query_masked_features)

        cross_entropy = cross_entropy_calc(pred, query_mask[:, 0, :, :])
#         loss = cross_entropy + 10*mse
        loss = cross_entropy
        loss.backward()
#         opt_Adam.step()
        optimizer.step()

#         print('Epoch %3d: %4d/%d Loss = 10*MSELoss + CrossEntropyLoss: %.4f = 10*%.4f + %.4f'%(epoch, i_iter+1, save_pred_every, loss.item(), mse.item(), cross_entropy.item()))
        print('Epoch %3d: %4d/%d Loss = %.6f'%(epoch, i_iter+1, save_pred_every, loss.item()))

        #save training loss
        accumulated_loss += loss.item()
#         accumulated_cross_entropy += cross_entropy.item()
#         accumulated_mse += mse.item()
#         if i_iter % save_pred_every == 0 and i_iter != 0:
    history_doc['train loss'].append(accumulated_loss / save_pred_every)
#     history_doc['train mse loss'].append(accumulated_mse / save_pred_every)
#     history_doc['train cross entropy loss'].append(accumulated_cross_entropy / save_pred_every)
        
    print ('----Evaluation----')
    with torch.no_grad():
        accumulated_loss = 0
#         accumulated_cross_entropy = 0
#         accumulated_cos_similarity = 0
        model = model.eval()
#         valset.history_mask_list=[None] * 1000
        all_inter, all_union, all_predict = [0] * 5, [0] * 5, [0] * 5
        for i_iter, batch_ in enumerate(valloader):

            query_img, query_mask, support_img, support_mask, sample_class, index = batch_
            
            query_img = query_img.cuda()
            support_img = support_img.cuda()
            support_mask = support_mask.cuda()
            query_mask = query_mask.cuda()  # change formation for crossentropy use
#             query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use

            pred = model(query_img, support_img, support_mask)
            pred = nn.functional.interpolate(pred, size=config.input_size, mode='bilinear', align_corners=True)  #upsample
#             pred = torch.squeeze(pred, 1)
#             support_masked_features = get_masked_features(support_feature_maps, support_mask)
#             query_masked_features = get_masked_features(query_feature_maps, query_mask)
#             val_cos_similarity = cos_similarity_calc(query_features, query_masked_features)
            
                
            query_mask = query_mask[:, 0, :, :]
#             print(query_mask.size())
#             val_cross_entropy = cross_entropy_calc(pred, query_mask)                
#             val_loss = val_cross_entropy + val_cos_similarity
            val_loss = cross_entropy_calc(pred, query_mask)
            accumulated_loss += val_loss.item()
#             accumulated_cross_entropy += val_cross_entropy.item()
#             accumulated_cos_similarity += val_cos_similarity.item()
#             print(torch.max(pred))
#             pred_softmax = F.softmax(pred, dim=1)
#             print(pred_softmax.size())
#             print(torch.max(pred_softmax))
            _, pred_label = torch.max(pred, 1)
#             print(pred_label.size())
            pred_label = pred_label.data.cpu()
            
#             pred_label[pred_label>0] = 1
#             pred_label[pred_label<0] = 0
#             pred_label = pred_label.long()
            inter_list, union_list, _, num_predict_list = get_iou(query_mask.cpu().long(), pred_label)
#             print(inter_list, union_list)
            
            for j in range(query_mask.shape[0]):# watch out last drop last!!(!=batch size)
                all_inter[sample_class[j] - (config.fold * 5 + 1)] += inter_list[j]
                all_union[sample_class[j] - (config.fold * 5 + 1)] += union_list[j]
        history_doc['val loss'].append(accumulated_loss/len(valloader))
#         history_doc['val cross entropy loss'].append(accumulated_cross_entropy/len(valloader))
#         history_doc['val cos similarity loss'].append(accumulated_cos_similarity/len(valloader))
        
        IOU = [0] * 5
        for j in range(5):
#             if all_union[j] != 0:
            print(all_union[j])
            IOU[j] = all_inter[j] / all_union[j]
#             else:
#                 IOU[j] = 0
            print('Category:', config.category[config.fold][j], IOU[j])
            history_doc[config.category[config.fold][j]].append(IOU[j])
                
        
        mean_iou = np.mean(IOU)
        
        history_doc['val iou'].append(mean_iou)
        print('Epoch: %d | IOU: %.4f | Learning rate: %.7f' % (epoch, mean_iou, optimizer.param_groups[0]['lr']))

#         iou_list.append(best_iou)
#         plot_iou(checkpoint_dir, iou_list)
#         np.savetxt(os.path.join(checkpoint_dir, 'iou_history.txt'), np.array(iou_list))
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
    
    
    writer.add_scalars('Loss', {'train': np.array(history_doc['train loss'][-1]), 'val': np.array(history_doc['val loss'][-1])}, epoch)
#     writer.add_scalar('train MSE Loss', np.array(history_doc['train mse loss'][-1]), epoch)
#     writer.add_scalar('train Cross Entropy Loss', np.array(history_doc['train cross entropy loss'][-1]), epoch)
#     writer.add_scalars('Cross Entropy Loss', 
#                        {'train': np.array(history_doc['train cross entropy loss'][-1]), 'val': np.array(history_doc['val cross entropy loss'][-1])}, epoch)
    writer.add_scalar('IoU', np.array(history_doc['val iou'][-1]), epoch)
    
    
    epoch_time = time.time() - begin_time
    pd.DataFrame(history_doc).to_csv(os.path.join(checkpoint_dir, 'train_log.csv'), index=False)
    print('Best epoch:%d ,iout:%.4f' % (best_epoch, best_iou))
    print('This epoch takes:', epoch_time/3600, 'hours')
    print('Still need %.4f hours' % ((config.num_epoch - epoch) * epoch_time / 3600))
    
writer.close()



