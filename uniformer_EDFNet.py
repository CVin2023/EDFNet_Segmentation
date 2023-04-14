import torch
import os
import torch.nn as nn
# from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
# from toolbox.models.H2.ResNet import Backbone_ResNet34_in3
import sys
from collections import OrderedDict
import functools
from toolbox.models.H2.uniformer import UniFormer
# __all__ = ['HD2S']

class Up(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(Up, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
# '''conv+bn+relu'''
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

'''
MAIN_MODEL!!!!
'''
class HD2S(nn.Module):
    def __init__(self, ):
        super(HD2S, self).__init__()
        # self.div_2,self.div_4,self.div_8,self.div_16,self.div_32=Backbone_ResNet34_in3()
        self.uniformer = UniFormer()
        self.fusion5 = Fusion1(in_plane1=512,in_plane2=256)
        self.fusion4 = Fusion1(in_plane1=512,in_plane2=320)
        self.fusion3 = Fusion1(in_plane1=320,in_plane2=128)
        self.fusion2 = Fusion1(in_plane1=128,in_plane2=64)
        self.fusion1 = RAttention(in_planes=64)

        self.conv3x3 = BasicConv2d(3, 64, kernel_size=3,stride=1,padding=1)
        self.conv64_3 = BasicConv2d(64, 3, kernel_size=3,stride=1,padding=1)
        self.conv256_128 = BasicConv2d(256, 128, kernel_size=3,stride=1,padding=1)
        self.conv128_64 = BasicConv2d(128, 64, kernel_size=3,stride=1,padding=1)

        self.upsample2 = Up(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = Up(scale_factor=4, mode='bilinear', align_corners=True)

        self.Decoder = Decoder(in_channel=64,out_channel=64)

        self.Edge = Edge()
        self.Merge_out1 = Merge_out(in1=320,in2=512)
        self.Merge_out2 = Merge_out(in1=128,in2=320)
        self.Merge_out3 = Merge_out(in1=64,in2=128)
        self.Merge_out4 = Merge_out(in1=64,in2=64)
        self.Merge_out5 = Merge_out(in1=64,in2=256)
        self.last = nn.Conv2d(64,3,3,1,1)

    def forward(self, x, y):
        merges = []

        rgb = self.uniformer(x)
        dep = self.uniformer(y)

        merge = self.fusion1(rgb[0],dep[0])
        merges.append(merge)
        merge = self.fusion2(rgb[1],dep[1],rgb[0],dep[0])
        merges.append(merge)
        merge = self.fusion3(rgb[2],dep[2],rgb[1],dep[1])
        merges.append(merge)
        merge = self.fusion4(rgb[3],dep[3],rgb[2],dep[2])
        merges.append(merge)
        edge_out = self.Edge(rgb[1],rgb[2])
        # #Merge_out
        merge_out1 = self.Merge_out1(merges[2],merges[3])
        merge_out2 = self.Merge_out2(merges[1],merge_out1)
        merge_out3 = self.Merge_out3(merges[0],merge_out2)
        # merge_out4 = self.Merge_out4(merges[0],merge_out3)

        merge_out2 = self.upsample4(merge_out2)
        merge_out2 = self.upsample2(merge_out2)
        merge_out2 = self.conv128_64(merge_out2)
        merge_out4 = self.upsample4(merge_out3)
        merge_out4 = torch.cat((merge_out2,merge_out4),dim=1)
        merge_out4 = self.conv128_64(merge_out4)


        out= self.Decoder(merge_out4,edge_out)
        out = self.last(out)

        edge_out = self.last(edge_out)

        return out,edge_out

class Edge(nn.Module):
    def __init__(self,):
        super(Edge,self).__init__()

        self.bcon1 = BasicConv2d(in_planes=320,out_planes=64,kernel_size=3,stride=1,padding=1)
        self.bcon2 = BasicConv2d(in_planes=128,out_planes=64,kernel_size=3,stride=1,padding=1)
        self.bconv = BasicConv2d(in_planes=128,out_planes=64,kernel_size=3,stride=1,padding=1)
        self.upsample2 = Up(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = Up(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = Up(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self,d2,d3):
        d2 = self.bcon2(d2)
        d3 = self.bcon1(d3)
        d3 = self.upsample2(d3)
        out = torch.cat((d2,d3),dim=1)
        out = self.upsample8(out)
        out = self.bconv(out)
        return out

class Merge_out(nn.Module):
    def __init__(self, in1,in2):
        super(Merge_out,self).__init__()
        self.bcon2 = BasicConv2d(in_planes=in2,out_planes=in1,kernel_size=3,stride=1,padding=1)
        self.bconv = BasicConv2d(in_planes=in1*2,out_planes=in1,kernel_size=1,stride=1,padding=0)
        self.upsample2 = Up(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self,d1,d5):
        # d1 = self.bcon1(d1)
        d5 = self.bcon2(d5)
        d5 = self.upsample2(d5)
        out = torch.cat((d1,d5),dim=1)
        out = self.bconv(out)
        return out


class CAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(CAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes,in_planes//16,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//16,in_planes,1,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = avg_out
        return self.sigmoid(out)
class SAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SAttention, self).__init__()
        self.conv_s = nn.Conv2d(2,1,kernel,padding=1,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv_s(x)
        out = self.sigmoid(x)
        return out

class RAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(RAttention, self).__init__()
        self.ca = CAttention(in_planes*2, reduction)
        self.ca1 = CAttention(in_planes,reduction)
        self.conv = BasicConv2d(in_planes=in_planes*2,out_planes=in_planes,kernel_size=1,stride=1,padding=0)
        self.sa = SAttention()
        self.ca_depth = CAttention(in_planes,reduction)

    def forward(self, rgb, dep):
        x = torch.cat((rgb,dep),dim=1)
        x = self.ca(x)
        x = self.conv(x)
        # x = self.sa(channel_weight)
        depth = self.ca1(dep)
        # x_depth = self.sa(depth)
        out = rgb + x * dep+depth*dep

        return out

'''
Fusion
'''
class Fusion1(nn.Module):
    def __init__(self, in_plane1,in_plane2,reduction=16, bn_momentum=0.0003):
        self.init__ = super(Fusion1, self).__init__()
        self.in_plane1 = in_plane1
        self.in_plane2 = in_plane2
        self.conv_down = BasicConv2d(in_plane2, in_plane1, kernel_size=3,stride=1,padding=1)
        self.conv_down2 = BasicConv2d(in_plane2*2, in_plane1, kernel_size=3,stride=1,padding=1)
        self.conv = BasicConv2d(in_plane1,in_plane1,kernel_size=3,stride=1,padding=1)
        self.conv_up2 = BasicConv2d(in_plane1*2,in_plane1,kernel_size=3,stride=1,padding=1)
        self.down2 = nn.AvgPool2d((2, 2), stride=2)
        self.up2 = Up(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU()
        self.attention = RAttention(in_planes=in_plane1)

    def forward(self, x1,y1,x2,y2):
        x2 = self.down2(x2)
        y2 = self.down2(y2)
        mid_b = torch.cat((x2,y2),dim=1)
        high = self.conv_down2(mid_b)

        x1 =self.conv(x1)
        y1 = self.conv(y1)
        low = torch.cat((x1,y1),dim=1)
        low = self.conv_up2(low)

        x = self.attention(low,high)

        merge_out = x

        return merge_out

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()
        self.deconv = nn.Conv2d(in_channel, out_channel, 1, 1,0)
        self.last_conv1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.last_conv2 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, low, high):
        high = self.deconv(high)
        out = torch.cat((low, high), dim=1)
        out = self.last_conv2(out)
        return out

if __name__ == '__main__':
    model = HD2S()

    left = torch.randn(6, 3, 224, 224)
    right = torch.randn(6, 3, 224, 224)
    out = model(left,right)

    print(out[0].shape)
    print(out[1].shape)
    # print(out[2].shape)
    # print(out[3].shape)