import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


#code of dilated convolution part is referenced from https://github.com/speedinghzl/Pytorch-Deeplab

affine_par = True


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self.Resblock(block, 64, layers[0], stride=1)
        self.layer2 = self.Resblock(block, 128, layers[1], stride=2)
        self.layer3 = self.Resblock(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self.Resblock(block, 512, layers[3], stride=1, dilation=4)
#         self.conv_compress_34 = nn.Sequential(
#             nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias = True),
#             nn.ReLU(),
#             nn.Dropout2d(p=0.5)
#         )
        self.conv_compress_23 = nn.Sequential(
            nn.Conv2d(in_channels=1024+512, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias = True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
#         self.intersection = self.Intersection_Module()
#         self.maxpooling = nn.MaxPool2d(kernel_size=7, stride=6, padding=1) # return [256*7*7]

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=256*2, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.skip1=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256+1, 256, kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.skip2=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.skip3=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.dilation_conv_0 = nn.Sequential(
            nn.Conv2d(256 , 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )
        self.dilation_conv_1 = nn.Sequential(
            nn.Conv2d(256 , 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )
        self.dilation_conv_6 = nn.Sequential(
            nn.Conv2d(256 , 256 , kernel_size=3, stride=1, padding=6,dilation=6, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.dilation_conv_12 = nn.Sequential(
            nn.Conv2d(256 , 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.dilation_conv_18 = nn.Sequential(
            nn.Conv2d(256 , 256 , kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.layer_out1=nn.Sequential(
            nn.Conv2d(1280, 256 , kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

        )
        self.layer_out2=nn.Conv2d(256, num_classes, kernel_size=1,stride=1,bias=True)
#         self.out = nn.CosineSimilarity(dim=1).cuda()
#         self.sigmoid = nn.sigmoid(dim=1).cuda()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def Resblock(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)
    
#     def Intersection_Module(self):
#         inter_module = nn.Sequential(
#             nn.Conv2d(in_channels=256*2, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True), 
#             nn.ReLU(), 
#             nn.Dropout2d(p=0.5),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True), 
#             nn.ReLU(), 
#             nn.Dropout2d(p=0.5),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.ReLU(),
#             nn.Dropout2d(p=0.5)
#         )
#         return inter_module
    
    

    def forward(self, query_img, support_img, support_mask):
        # important: do not optimize the RESNET backbone
        query_img = self.conv1(query_img)
        query_img = self.bn1(query_img)
        query_img = self.relu(query_img)
        query_img = self.maxpool(query_img)
        query_feature1 = self.layer1(query_img)
        query_feature2 = self.layer2(query_feature1)
        query_feature3 = self.layer3(query_feature2)
        query_feature4 = self.layer4(query_feature3)
        query_feature_maps_23=torch.cat([query_feature2, query_feature3],dim=1)
#         query_feature_maps_34=torch.cat([query_feature3, query_feature4],dim=1)
        query_feature_maps_23=self.conv_compress_23(query_feature_maps_23)
#         query_feature_maps_4=self.conv_compress_34(query_feature4)
#         query_features = F.avg_pool2d(query_feature_maps, query_feature_maps.shape[-2:])

        support_img = self.conv1(support_img)
        support_img = self.bn1(support_img)
        support_img = self.relu(support_img)
        support_img = self.maxpool(support_img)
        support_feature1 = self.layer1(support_img)
        support_feature2 = self.layer2(support_feature1)
        support_feature3 = self.layer3(support_feature2)
        support_feature4 = self.layer4(support_feature3)
        support_feature_maps_23=torch.cat([support_feature2, support_feature3],dim=1)
#         support_feature_maps_34=torch.cat([support_feature3, support_feature4],dim=1)
        support_feature_maps_23=self.conv_compress_23(support_feature_maps_23)
#         support_feature_maps_4=self.conv_compress_34(support_feature4)
#         support_features = F.avg_pool2d(support_feature_maps, support_feature_maps.shape[-2:])
        
#         feature_size = query_rgb.shape[-2:]
#         support_rgb = self.layer5(support_rgb)
        support_mask = F.interpolate(support_mask, support_feature_maps_23.shape[-2:], mode='bilinear',align_corners=True)
        h, w = support_feature_maps_23.shape[-2:][0],support_feature_maps_23.shape[-2:][1]
        area = F.avg_pool2d(support_mask, [h,w]) * h * w + 0.0005
        
        support_conv = support_mask * support_feature4
        
        support_feature_maps_23 = support_mask * support_feature_maps_23
        support_features = F.avg_pool2d(input=support_feature_maps_23, kernel_size=[h,w]) * h * w / area
        support_features = support_features.expand(-1,-1, h, w)

#         support_conv = self.maxpooling(support_features)
#         print('support_conv: ', support_conv.shape)
        pre_mask = torch.tensor([]).cuda()
        for i in range(query_feature_maps_23.shape[0]):
            input_ = query_feature4[i]
            weight_ = support_conv[i]
#             print('input_: ', input_.shape, 'weight_: ', weight_.shape)
            input_ = input_.unsqueeze(0)
            weight_ = weight_.unsqueeze(0)
#             print('input_: ', input_.shape, 'weight_: ', weight_.shape)
            pre_mask = torch.cat([pre_mask, F.conv2d(input=input_, weight=weight_, stride=1, padding=20, groups=1)])
#         pre_mask = torch.tensor(pre_mask)
#         print('pre_mask: ', pre_mask.shape)
#         inter_features = self.intersection(torch.cat([query_features, support_features], dim=1))
#         inter_feature_maps = support_conv*query_feature_maps
#         inter_feature_maps = inter_features.expand(-1, -1, query_feature_maps.shape[-2:][0], query_feature_maps.shape[-2:][1])

        feature_maps = self.localization(torch.cat([query_feature_maps_23, support_features],dim=1))
#         feature_maps = self.localization(torch.cat([query_feature_maps_23, support_features],dim=1))
        
        feature_maps = feature_maps + self.skip1(torch.cat([feature_maps, pre_mask],dim=1))
        feature_maps = feature_maps + self.skip2(feature_maps)
        feature_maps = feature_maps + self.skip3(feature_maps)
        global_feature_maps = F.avg_pool2d(feature_maps, kernel_size=feature_maps.shape[-2:])
        global_feature_maps = self.dilation_conv_0(global_feature_maps)
        global_feature_maps = global_feature_maps.expand(-1,-1, feature_maps.shape[-2:][0], feature_maps.shape[-2:][1])
        
#         seg = torch.cat([global_feature_maps, 
        seg = torch.cat([global_feature_maps,
                   self.dilation_conv_1(feature_maps), 
                   self.dilation_conv_6(feature_maps), 
                   self.dilation_conv_12(feature_maps), 
                   self.dilation_conv_18(feature_maps)], dim=1)
        seg = self.layer_out1(seg)
        seg = self.layer_out2(seg)
#         seg = self.out(query_feature_maps, inter_feature_maps)
#         seg = torch.unsqueeze(seg, 1)
#         seg = torch.cat([1-seg, seg], dim=1)
        return seg #, inter_features, query_feature_maps




def network():
    model = ResNet(Bottleneck,[3, 4, 6, 3], 2)
    return model



if __name__ == '__main__':
    import torchvision

    def load_resnet_param(model, stop_layer='layer4'):
        resnet = torchvision.models.resnet50(pretrained=True)
        saved_state_dict = resnet.state_dict()
        new_params = model.state_dict().copy()

        for i in saved_state_dict:  # copy params from resnet50,except layers after stop_layer

            i_parts = i.split('.')

            if not i_parts[0] == stop_layer:

                new_params['.'.join(i_parts)] = saved_state_dict[i]
            else:
                break
        model.load_state_dict(new_params)
        model.train()
        return model

    model = network().cuda()
    model=load_resnet_param(model)

    query_rgb = torch.FloatTensor(1,3,321,321).cuda()
    support_rgb = torch.FloatTensor(1,3,321,321).cuda()
    support_mask = torch.FloatTensor(1,1,321,321).cuda()

#     history_mask=(torch.zeros(1,2,50,50)).cuda()

    seg = model(query_rgb,support_rgb, support_mask)
    print(model)
    print(seg.size())
    print(seg)






