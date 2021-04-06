import torchvision
import os
import torch
from pylab import plt
import torch.nn.functional as F
import csv
import numpy as np

def load_resnet_param(model, stop_layer='layer4', layer_num=50):
    if layer_num == 50:
        resnet = torchvision.models.resnet50(pretrained=True)
    else:
        resnet = torchvision.models.resnet101(pretrained=True)
    saved_state_dict = resnet.state_dict()
    new_params = model.state_dict().copy()

    for i in saved_state_dict:  # copy params from resnet101,except layers after stop_layer

        i_parts = i.split('.')

        if not i_parts[0] == stop_layer:

            new_params['.'.join(i_parts)] = saved_state_dict[i]
        else:
            break
    #         if i_parts[0] == stop_layer:
    #             new_params['.'.join(i_parts)] = saved_state_dict[i]

    model.load_state_dict(new_params)
    model.train()
    return model


def optim_or_not(model, yes):
    for param in model.parameters():
        if yes:
            param.requires_grad = True
        else:
            param.requires_grad = False


def turn_off(model, is_layer4_available=False):
    optim_or_not(model.module.conv1, False)
    optim_or_not(model.module.layer1, False)
    optim_or_not(model.module.layer2, False)
    optim_or_not(model.module.layer3, False)
    if is_layer4_available:
        optim_or_not(model.module.layer4, False)
        
def turn_on(model, is_layer4_available=False):
    optim_or_not(model.module.conv1, True)
    optim_or_not(model.module.layer1, True)
    optim_or_not(model.module.layer2, True)
    optim_or_not(model.module.layer3, True)
    if is_layer4_available:
        optim_or_not(model.module.layer4, True)


def get_lr_params(model):
    """
    get layers for optimization
    """
    b = []
    b.append(model.module.channel_compress.parameters())
    b.append(model.module.IEM.parameters())
    b.append(model.module.layer5.parameters())
    b.append(model.module.skip1.parameters())
    b.append(model.module.skip2.parameters())
    b.append(model.module.skip3.parameters())
    b.append(model.module.dilation_conv_0.parameters())
    b.append(model.module.dilation_conv_1.parameters())
    b.append(model.module.dilation_conv_6.parameters())
    b.append(model.module.dilation_conv_12.parameters())
    b.append(model.module.dilation_conv_18.parameters())
    b.append(model.module.layer_out1.parameters())
    b.append(model.module.decoder.parameters())
    b.append(model.module.layer_out2.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i



def cross_entropy_calc(pred, label, weight=None):
    label = label.long()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
    #     criterion = torch.nn.CrossEntropyLoss()
    #     criterion = torch.nn.BCEWithLogitsLoss().cuda()
    return criterion(pred, label)

def cos_similarity_calc(pred, label):
    criterion = torch.nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    return torch.sum(criterion(pred, label))

def mse_calc(pred, label):
    criterion = torch.nn.MSELoss(reduction='mean')
    return criterion(pred, label)

def get_iou(query_mask, pred_label, mode='foreground'):  # pytorch 1.0 version
    if mode == 'background':
        query_mask = 1 - query_mask
        pred_label = 1 - pred_label
    num_img = query_mask.shape[0]  # batch size
    num_predict_list, inter_list, union_list, iou_list = [], [], [], []
    for i in range(num_img):
        num_predict = torch.sum((pred_label[i] > 0).float()).item()
        combination = (query_mask[i] + pred_label[i]).float()
        inter = torch.sum((combination == 2).float()).item()
        union = torch.sum((combination == 1).float()).item() + torch.sum((combination == 2).float()).item()
        if union != 0:
            inter_list.append(inter)
            union_list.append(union)
            iou_list.append(inter / union)
            num_predict_list.append(num_predict)
        else:
            inter_list.append(inter)
            union_list.append(union)
            iou_list.append(0)
            num_predict_list.append(num_predict)
    return inter_list, union_list, iou_list, num_predict_list


def adjust_learning_rate(optimizer, lr, epoch, reduce_per_epoch=20):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1 ** (epoch // reduce_per_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def wirte_csv(doc_list, path):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(doc_list)

def write_pairs2txt(path, support, query, sample_class):
    file = open(path,"w")
    for i in range(len(support)):
        file.write(support[i]+' '+query[i]+' '+str(sample_class[i]+1)+'\n')
#         file.write('\n')
    file.close()
    
def convert_2d(r):
    # 添加均值为 0, 标准差为 64 的加性高斯白噪声
    s = r + np.random.normal(0, 64, r.shape)
    if np.min(s) >= 0 and np.max(s) <= 255:
        return s
    # 对比拉伸
#     s = s - np.full(s.shape, np.min(s))
#     s = s * 255 / np.max(s)
    s[s>255]=255
    s[s<0]=0
    s = s.astype(np.uint8)
    return s