import torch
import torch.nn as nn
import torch.nn.functional as F
from model.EfficientNet import EfficientNet
from util.effi_utils import get_model_shape
from modules.att_modules import RFB_Block, aggregation, SEA_Block


class RAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = EfficientNet.from_pretrained(f'efficientnet-b{cfg.arch}', advprop=True)
        self.block_idx, self.channels = get_model_shape()
        self.H = cfg.img_size
        self.W = cfg.img_size

        # Receptive Field Blocks
        channels = [int(arg_c) for arg_c in cfg.RFB_aggregated_channel]
        self.rfb2 = RFB_Block(self.channels[1], channels[0])
        self.rfb3 = RFB_Block(self.channels[2], channels[1])
        self.rfb4 = RFB_Block(self.channels[3], channels[2])
        self.rfb4_2 = RFB_Block(self.channels[3], 32)
        self.conv = nn.Conv2d(32, 8, kernel_size=3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), bias=False)

        self.sae2 = SEA_Block(32, 64)
        self.sae3 = SEA_Block(64, 128)

        # Multi-level aggregation
        self.agg = aggregation(channels)

        self.con_edge5 = nn.Conv2d(128, 64, 3, padding=1)
        self.con_edge4 = nn.Conv2d(64, 32, 3, padding=1)
        self.con_edge = nn.Conv2d(32, 1, 3, padding=1)
        self.con_edge_2 = nn.Conv2d(2, 1, 3, padding=1)
        self.sae1 = SEA_Block(32, 32)


    def forward(self, inputs):
        B, C, H, W = inputs.size()

        # EfficientNet backbone Encoder
        x = self.model.initial_conv(inputs)
        features = self.model.get_blocks(x, H, W)

        edge = features[0]

        x3_rfb = self.rfb2(features[1])
        x4_rfb = self.rfb3(features[2])
        x5_rfb = self.rfb4(features[3])

        edge5 = F.interpolate(x5_rfb, scale_factor=2, mode='bilinear')
        edge5 = self.con_edge5(edge5)
        edge4 = torch.cat((x4_rfb, edge5), 1)
        edge4 = self.con_edge5(edge4)

        edge4 = F.interpolate(edge4, scale_factor=2, mode='bilinear')
        edge4 = self.con_edge4(edge4)
        edge3 = torch.cat((x3_rfb, edge4), 1)
        edge3 = self.con_edge4(edge3)

        edge3 = F.interpolate(edge3, scale_factor=2, mode='bilinear')
        edge = torch.cat((edge, edge3), 1)
        edge = self.con_edge4(edge)
        
        edge = torch.relu(self.conv(edge))
        edge = F.interpolate(edge, size=(self.H, self.W), mode='bilinear')
        

        x4_sea = self.sae2(x4_rfb, x3_rfb)
        x5_sea = self.sae3(x5_rfb, x4_rfb)

        D_0 = self.agg(x5_sea, x4_sea, x3_rfb)
        ds_map0 = F.interpolate(D_0, scale_factor=8, mode='bilinear')

        return torch.sigmoid(ds_map0), torch.sigmoid(edge)