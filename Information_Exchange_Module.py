import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3)
        return scale
    
class co_excitation_block(nn.Module):
    def __init__(self, inplanes):
        super(co_excitation_block, self).__init__()

        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = self.in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.Q = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.Q[1].weight, 0)
        nn.init.constant_(self.Q[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

#         self.concat_project = nn.Sequential(
#             nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
#             nn.ReLU()
#         )
        
        self.ChannelGate = ChannelGate(self.in_channels)
#         self.globalAvgPool = nn.AdaptiveAvgPool2d(1)


        
    def forward(self, q, s):

        

        batch_size, channels, height_q, width_q = q.shape
        batch_size, channels, height_s, width_s = s.shape


        #####################################find aim image similar object ####################################################

        q_x = self.g(q).view(batch_size, self.inter_channels, -1)
        q_x = q_x.permute(0, 2, 1)

        s_x = self.g(s).view(batch_size, self.inter_channels, -1)
        s_x = s_x.permute(0, 2, 1)

        theta_x = self.theta(s).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(q).view(batch_size, self.inter_channels, -1)

        

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N
#         f_div_C = F.softmax(f, dim=-1)

        f = f.permute(0, 2, 1).contiguous()
        N = f.size(-1)
        fi_div_C = f / N
#         fi_div_C = F.softmax(f, dim=-1)

        non_s = torch.matmul(f_div_C, q_x)
        non_s = non_s.permute(0, 2, 1).contiguous()
        non_s = non_s.view(batch_size, self.inter_channels, height_s, width_s)
        non_s = self.W(non_s)
        non_s = non_s + s

        non_q = torch.matmul(fi_div_C, s_x)
        non_q = non_q.permute(0, 2, 1).contiguous()
        non_q = non_q.view(batch_size, self.inter_channels, height_q, width_q)
        non_q = self.Q(non_q)
        non_q = non_q + q

        ##################################### Response in chaneel weight ####################################################

        c_weight = self.ChannelGate(s)
        act_s = non_s * c_weight
        act_q = non_q * c_weight

        return act_q, act_s
    
def build_co_excitation_block(inplanes):
    return co_excitation_block(inplanes)

def build_channel_gate_block(inplanes):
    return ChannelGate(inplanes)