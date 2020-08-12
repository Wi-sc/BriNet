import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from decoder_module import build_decoder
# from nonlocal_block import build_nonlocal_block
from Information_Exchange_Module import build_co_excitation_block
# from co_excitation_block import build_channel_gate_block

# code of dilated convolution part is referenced from https://github.com/speedinghzl/Pytorch-Deeplab

affine_par = True


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
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
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
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
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
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
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.Resblock(block, 64, layers[0], stride=1)
        self.layer2 = self.Resblock(block, 128, layers[1], stride=2)
        self.layer3 = self.Resblock(block, 256, layers[2], stride=1, dilation=2)
        self.channel_compress = nn.Sequential(
            nn.Conv2d(in_channels=1024+512, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.co_excitation_block = build_co_excitation_block(256)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256+256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.skip1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.skip2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.skip3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.dilation_conv_0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.dilation_conv_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.dilation_conv_6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.dilation_conv_12 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.dilation_conv_18 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.layer_out1 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

        )
        self.layer_out2 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self.decoder = build_decoder(num_classes, 256, nn.BatchNorm2d)
        
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
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, query_img, support_img, support_mask):
        # important: do not optimize the RESNET backbone
        query_img = self.conv1(query_img)
        query_img = self.bn1(query_img)
        query_img = self.relu(query_img)
        query_img = self.maxpool(query_img)
        query_feature1 = self.layer1(query_img)
        query_feature2 = self.layer2(query_feature1)
        query_feature3 = self.layer3(query_feature2)
        query_feature_maps_23 = torch.cat([query_feature2, query_feature3], dim=1)
        query_feature_maps = self.channel_compress(query_feature_maps_23)
        
        support_img = self.conv1(support_img)
        support_img = self.bn1(support_img)
        support_img = self.relu(support_img)
        support_img = self.maxpool(support_img)
        support_feature1 = self.layer1(support_img)
        support_feature2 = self.layer2(support_feature1)
        support_feature3 = self.layer3(support_feature2)
        support_feature_maps_23 = torch.cat([support_feature2, support_feature3], dim=1)
        support_feature_maps = self.channel_compress(support_feature_maps_23)
        
        batch, channel, h, w = support_feature_maps.shape[:]
        _, _, qh, qw = query_feature_maps.shape[:]
        support_mask = F.interpolate(support_mask, support_feature_maps.shape[-2:], mode='bilinear', align_corners=True)
        support_feature_maps_masked = support_mask * support_feature_maps
        area = F.avg_pool2d(support_mask, [h, w]) * h * w + 5e-5

        query_feature_maps, support_feature_maps_masked = self.co_excitation_block(query_feature_maps, support_feature_maps_masked)
        
        sup_conv_1 = F.avg_pool2d(support_feature_maps_masked, [h, w])
        sup_conv_1 = sup_conv_1.expand(-1, -1, query_feature_maps.shape[-2], query_feature_maps.shape[-1])
        correlation_feature_map = query_feature_maps*sup_conv_1
        
        query_feature_maps = query_feature_maps.view(1, batch*channel, qh, qw)

        sup_conv_13 = F.adaptive_avg_pool2d(support_feature_maps_masked, output_size=[1,3])
        sup_conv_13 = sup_conv_13.view(batch*channel, 1, 1, 3)
        correlation_feature_map_1 = F.conv2d(input=query_feature_maps, weight=sup_conv_13, stride=1, padding=(0, 1), groups=batch*channel)
        correlation_feature_map += correlation_feature_map_1.view(batch, channel, qh, qw)
        
        sup_conv_31 = F.adaptive_avg_pool2d(support_feature_maps_masked, output_size=[3, 1])
        sup_conv_31 = sup_conv_31.view(batch*channel, 1, 3, 1)
        correlation_feature_map_2 = F.conv2d(input=query_feature_maps, weight=sup_conv_31, stride=1, padding=(1, 0), groups=batch*channel)
        correlation_feature_map += correlation_feature_map_2.view(batch, channel, qh, qw)

        query_feature_maps = query_feature_maps.view(batch,channel,qh,qw) 

        feature_maps = self.layer5(torch.cat([correlation_feature_map, query_feature_maps], dim=1))
        feature_maps = feature_maps + self.skip1(feature_maps)
        feature_maps = feature_maps + self.skip2(feature_maps)
        feature_maps = feature_maps + self.skip3(feature_maps)

        global_feature_maps = F.avg_pool2d(feature_maps, kernel_size=feature_maps.shape[-2:])
        global_feature_maps = self.dilation_conv_0(global_feature_maps)
        global_feature_maps = global_feature_maps.expand(-1, -1, feature_maps.shape[-2:][0], feature_maps.shape[-2:][1])

        aspp_feature = torch.cat([global_feature_maps,
                         self.dilation_conv_1(feature_maps),
                         self.dilation_conv_6(feature_maps),
                         self.dilation_conv_12(feature_maps),
                         self.dilation_conv_18(feature_maps)], dim=1)
        aspp_feature = self.layer_out1(aspp_feature)
        final_mask = self.decoder(aspp_feature, query_feature1)
#         final_mask = self.layer_out2(aspp_feature)
        
        if self.training:
            aux_mask = self.layer_out2(aspp_feature)
            return final_mask, aux_mask
        else:
            return final_mask

def network():
    model = ResNet(Bottleneck, [3, 4, 6, 3], 2)
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
    model = load_resnet_param(model)

    query_img = torch.FloatTensor(2, 3, 353, 353).cuda()
    support_img = torch.FloatTensor(2, 3, 353, 353).cuda()
    support_mask = torch.FloatTensor(2, 1, 353, 353).cuda()

#     print(model)
    final_mask, aux_mask = model(query_img, support_img, support_mask)
    print(final_mask.size(), aux_mask.size())