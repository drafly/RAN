"""
author: Min Seok Lee and Wooseok Shin
"""
import torch.nn as nn
import torch.nn.functional as F
from config import getConfig

cfg = getConfig()


class RFB_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_Block, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x


class MS_CAM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        wei = self.sigmoid(xl)

        return wei


class SEA_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SEA_Block, self).__init__()
        self.conv_x2 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.conv = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channel*2, out_channel, 3, padding=1)
        self.channel = out_channel
        self.SA = MS_CAM(out_channel)

    def forward(self, x1, x2):
        x1_1 = x1
        x1_2 = x1
        x1_size = x1.size()[2:]
        x2 = self.conv_x2(x2)
        x2 = F.interpolate(x2, x1_size, mode='bilinear', align_corners=True)
        w1 = torch.sigmoid(x2)
        w2 = torch.sigmoid(-x2)
        x1_1 = w1.mul(x1_1)
        x1_2 = w2.mul(x1_2)

        x1_1 = self.conv(x1_1)
        x1_2 = self.conv(x1_2)

        x3 = x1_1 + x1_2
        w = self.SA(x3)
        
        y = x1_1.mul(1-w) + x1_2.mul(w)
        return y

class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = nn.Conv2d(channel[2], channel[1], 3, padding=1)
        self.conv_upsample2 = nn.Conv2d(channel[2], channel[0], 3, padding=1)
        self.conv_upsample3 = nn.Conv2d(channel[1], channel[0], 3, padding=1)
        self.conv_upsample4 = nn.Conv2d(channel[2], channel[2], 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(channel[2] + channel[1], channel[2] + channel[1], 3, padding=1)

        self.conv_concat2 = nn.Conv2d((channel[2] + channel[1]), (channel[2] + channel[1]), 3, padding=1)
        self.conv_concat3 = nn.Conv2d((channel[0] + channel[1] + channel[2]),
                                        (channel[0] + channel[1] + channel[2]), 3, padding=1)
        self.conv_upsample6 = nn.Conv2d((channel[0] + channel[1] + channel[2]), 1, 3, padding=1)

        # self.UAM = UnionAttentionModule(channel[0] + channel[1] + channel[2])

    def forward(self, e4, e3, e2):
        e4_1 = e4
        e3_1 = self.conv_upsample1(self.upsample(e4)) * e3
        e2_1 = self.conv_upsample2(self.upsample(self.upsample(e4))) \
               * self.conv_upsample3(self.upsample(e3)) * e2

        e3_2 = torch.cat((e3_1, self.conv_upsample4(self.upsample(e4_1))), 1)
        e3_2 = self.conv_concat2(e3_2)

        e2_2 = torch.cat((e2_1, self.conv_upsample5(self.upsample(e3_2))), 1)
        x = self.conv_concat3(e2_2)

        # output = self.UAM(x)
        output = self.conv_upsample6(x)

        return output