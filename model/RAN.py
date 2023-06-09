import torch
import torch.nn as nn
import torch.nn.functional as F
from model.EfficientNet import EfficientNet
from util.effi_utils import get_model_shape
from modules.att_modules import RFB_Block, aggregation, RSA_Block


class RAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = EfficientNet.from_pretrained(f'efficientnet-b3', advprop=True)
        self.block_idx, self.channels = get_model_shape()
        self.H = cfg.img_size
        self.W = cfg.img_size

        # Receptive Field Blocks
        RFB_aggregated_channel = [32, 64, 128]
        channels = [int(arg_c) for arg_c in RFB_aggregated_channel]
        self.rfb2 = RFB_Block(self.channels[1], channels[0])
        self.rfb3 = RFB_Block(self.channels[2], channels[1])
        self.rfb4 = RFB_Block(self.channels[3], channels[2])

        # eight-channel connectivity contour
        self.conv = nn.Conv2d(32, 8, kernel_size=3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), bias=False)
        self.con_edge4 = nn.Conv2d(128, 64, 3, padding=1)
        self.con_edge3 = nn.Conv2d(64, 32, 3, padding=1)

        # region separation attention
        self.rsa2 = RSA_Block(32, 64)
        self.rsa3 = RSA_Block(64, 128)
        

        # Multi-level feature aggregation
        self.agg = aggregation(channels)
        


    def forward(self, inputs):
        _, _, H, W = inputs.size()

        # EfficientNet backbone Encoder
        x = self.model.initial_conv(inputs)
        features = self.model.get_blocks(x, H, W)

        x2_rfb = self.rfb2(features[1])
        x3_rfb = self.rfb3(features[2])
        x4_rfb = self.rfb4(features[3])

        contour4 = F.interpolate(x4_rfb, scale_factor=2, mode='bilinear')
        contour4 = self.con_edge4(contour4)
        contour3 = torch.cat((x3_rfb, contour4), 1)
        contour3 = self.con_edge4(contour3)

        contour3 = F.interpolate(contour3, scale_factor=2, mode='bilinear')
        contour3 = self.con_edge3(contour3)
        contour2 = torch.cat((x2_rfb, contour3), 1)
        contour2 = self.con_edge3(contour2)

        contour = features[0]
        contour2 = F.interpolate(contour2, scale_factor=2, mode='bilinear')
        C = torch.cat((contour, contour2), 1)
        C = self.con_edge3(C)
        
        C = torch.relu(self.conv(C))
        C = F.interpolate(C, size=(self.H, self.W), mode='bilinear')
        

        x3_rsa = self.rsa2(x3_rfb, x2_rfb)
        x4_rsa = self.rsa3(x4_rfb, x3_rfb)

        D = self.agg(x4_rsa, x3_rsa, x2_rfb)
        D = F.interpolate(D, scale_factor=8, mode='bilinear')

        return torch.sigmoid(D), torch.sigmoid(C)