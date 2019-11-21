import torchvision
import os
import torch
from pylab import plt
import torch.nn.functional as F


def load_resnet_param(model, stop_layer='layer4', layer_num=50):
    if layer_num==50:
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
    model.load_state_dict(new_params)
    model.train()
    return model




def optim_or_not(model, yes):
    for param in model.parameters():
        if yes:
            param.requires_grad = True
        else:
            param.requires_grad = False


def turn_off(model):
    optim_or_not(model.module.conv1, False)
    optim_or_not(model.module.layer1, False)
    optim_or_not(model.module.layer2, False)
    optim_or_not(model.module.layer3, False)


def get_10x_lr_params(model):
    """
    get layers for optimization
    """
    b = []
    b.append(model.module.conv_compress_23.parameters())
#     b.append(model.module.conv_compress_34.parameters())
#     b.append(model.module.intersection.parameters())
#     b.append(model.module.ConvLSTM.parameters())
    b.append(model.module.localization.parameters())
    b.append(model.module.skip1.parameters())
    b.append(model.module.skip2.parameters())
    b.append(model.module.skip3.parameters())
    b.append(model.module.dilation_conv_0.parameters())
    b.append(model.module.dilation_conv_1.parameters())
    b.append(model.module.dilation_conv_6.parameters())
    b.append(model.module.dilation_conv_12.parameters())
    b.append(model.module.dilation_conv_18.parameters())
    b.append(model.module.layer_out1.parameters())
    b.append(model.module.layer_out2.parameters())
    
#     b.append(model.module.out.parameters())
    
#     b.append(model.module.layer5_1.parameters())
#     b.append(model.module.layer5_2.parameters())
#     b.append(model.module.layer5_3.parameters())
#     b.append(model.module.layer6_1.parameters())
#     b.append(model.module.layer6_2.parameters())
#     b.append(model.module.layer6_3.parameters())
#     b.append(model.module.segmentation_module_1.parameters())
#     b.append(model.module.segmentation_module_2.parameters())
#     b.append(model.module.segmentation_module_3.parameters())

    # benchmark
#     b = []
#     b.append(model.module.layer5.parameters())
#     b.append(model.module.layer55.parameters())
#     b.append(model.module.layer6_0.parameters())
#     b.append(model.module.layer6_1.parameters())
#     b.append(model.module.layer6_2.parameters())
#     b.append(model.module.layer6_3.parameters())
#     b.append(model.module.layer6_4.parameters())
#     b.append(model.module.layer7.parameters())
#     b.append(model.module.layer9.parameters())
#     b.append(model.module.residule1.parameters())
#     b.append(model.module.residule2.parameters())
#     b.append(model.module.residule3.parameters())
    
    for j in range(len(b)):
        for i in b[j]:
            yield i

def get_adam_params(model):
    """
    get layers for optimization
    """
    b = []
#     b.append(model.module.conv_compress.parameters())
#     b.append(model.module.intersection.parameters())
    b.append(model.module.ConvLSTM.parameters())
    
    for j in range(len(b)):
        for i in b[j]:
            yield i




def cross_entropy_calc(pred, label):
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

def plot_loss(checkpoint_dir,loss_list,save_pred_every):
    x=range(0,len(loss_list)*save_pred_every,save_pred_every)
    y=loss_list
    plt.switch_backend('agg')
    plt.plot(x,y,color='blue',marker='o',label='Train loss')
    plt.xticks(range(0,len(loss_list)*save_pred_every+3,(len(loss_list)*save_pred_every+10)//10))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir,'loss_fig.pdf'))
    plt.close()


def plot_iou(checkpoint_dir,iou_list):
    x=range(0,len(iou_list))
    y=iou_list
    plt.switch_backend('agg')
    plt.plot(x,y,color='red',marker='o',label='IOU')
    plt.xticks(range(0,len(iou_list)+3,(len(iou_list)+10)//10))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir,'iou_fig.pdf'))
    plt.close()



def get_iou(query_mask, pred_label ,mode='foreground'):#pytorch 1.0 version
    if mode=='background':
        query_mask=1-query_mask
        pred_label=1-pred_label
    num_img=query_mask.shape[0]#batch size
    num_predict_list,inter_list, union_list, iou_list=[],[],[],[]
    for i in range(num_img):
        num_predict=torch.sum((pred_label[i]>0).float()).item()
        combination = (query_mask[i] + pred_label[i]).float()
        inter = torch.sum((combination == 2).float()).item()
        union = torch.sum((combination == 1).float()).item()+torch.sum((combination ==2).float()).item()
        if union!=0:
            inter_list.append(inter)
            union_list.append(union)
            iou_list.append(inter/union)
            num_predict_list.append(num_predict)
        else:
            inter_list.append(inter)
            union_list.append(union)
            iou_list.append(0)
            num_predict_list.append(num_predict)
    return inter_list,union_list,iou_list,num_predict_list

def get_masked_features(feature_map, mask):
    mask = F.interpolate(mask, feature_map.shape[-2:], mode='bilinear',align_corners=True)
    h, w = feature_map.shape[-2:][0], feature_map.shape[-2:][1]
    area = F.avg_pool2d(mask, feature_map.shape[-2:]) * h * w + 0.0005
    masked_features = mask * feature_map
    masked_features = F.avg_pool2d(input=masked_features, kernel_size=feature_map.shape[-2:]) * h * w / area
    return masked_features