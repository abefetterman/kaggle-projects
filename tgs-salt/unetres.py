import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import (
    BasicBlock,
    ResNet,
    resnet50,
)

from nl import NONLocalBlock2D

class ConvBn2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
            padding=padding, stride=1)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x)

class cSELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(cSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ELU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class sSELayer(nn.Module):
    def __init__(self, channel):
        super(sSELayer, self).__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        return x * self.sigmoid(y)

class scSELayer(nn.Module):
    def __init__(self, channels):
        super(scSELayer, self).__init__()
        self.sSE = sSELayer(channels)
        self.cSE = cSELayer(channels)
    def forward(self, x):
        a = self.sSE(x)
        b = self.cSE(x)
        return a + b

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels, upsample=True):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.se = scSELayer(out_channels)

    def forward(self, x, e=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return self.se(x)

class NonLocalBlock(nn.Module):
    def __init__(self, channels, upsample=True):
        super().__init__()
        self.nl = NONLocalBlock2D(channels)
        self.upsample = upsample
    def forward(self, x, e=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)

        x = self.nl(x)
        return x

class UnetModel(nn.Module):
    def __init__(self, decoder_channels=128):
        super().__init__()
        # self.resnet = ResNet(BasicBlock, [3,4,6,3], num_classes=1)
        self.resnet = resnet50(pretrained=True)
        self.expansion = 4

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        self.se2 = scSELayer(64*self.expansion)
        self.se3 = scSELayer(128*self.expansion)
        self.se4 = scSELayer(256*self.expansion)
        self.se5 = scSELayer(512*self.expansion)

        self.center = nn.Sequential(
            ConvBn2d(512*self.expansion, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512*self.expansion, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, decoder_channels),
            nn.ELU(inplace=True),
        )
        self.empty_predict = nn.Linear(decoder_channels, 1)

        self.decoder5 = Decoder(256+512*self.expansion, 512, decoder_channels, upsample=False)
        self.decoder4 = Decoder( decoder_channels+256*self.expansion, 256, decoder_channels)
        self.decoder3 = Decoder( decoder_channels+128*self.expansion, 128, decoder_channels)
        self.decoder2 = Decoder( decoder_channels+ 64*self.expansion,  64, decoder_channels)
        self.decoder1 = Decoder( decoder_channels    ,  32, decoder_channels)

        self.logit = nn.Sequential(
            nn.Conv2d(decoder_channels*6, decoder_channels, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(decoder_channels, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        batch_size = x.shape[0]
        in_px = x.shape[-1]
        depth = torch.linspace(0,1,in_px).repeat([batch_size,1,in_px,1]).to(x.device)
        x = torch.cat([
            (x[:,2:]-mean[2])/std[2],
            (depth-mean[1])/std[1],
            (depth*x[:,0:1]-mean[0])/std[0],
            # (x[:,1:2]-mean[1])/std[1],
            # (x[:,0:1]-mean[0])/std[0],
        ], 1)

        x = self.conv1(x)
        e2 = self.se2(self.encoder2(x))
        e3 = self.se3(self.encoder3(e2))
        e4 = self.se4(self.encoder4(e3))
        e5 = self.se5(self.encoder5(e4))
        # e2 = self.encoder2(x)
        # e3 = self.encoder3(e2)
        # e4 = self.encoder4(e3)
        # e5 = self.encoder5(e4)

        f = self.center(e5)
        ipool = self.avgpool(e5).view(batch_size,-1)
        ipool = self.fc(ipool)

        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)

        f = torch.cat([
            d1,
            F.interpolate(d2,scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3,scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4,scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5,scale_factor=16, mode='bilinear', align_corners=False),
            F.interpolate(ipool.view(batch_size,-1,1,1), scale_factor=in_px, mode='bilinear', align_corners=False)
        ], 1)

        deep_layers = torch.cat([
            d1[:,:1],
            F.interpolate(d2[:,:1],scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3[:,:1],scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4[:,:1],scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5[:,:1],scale_factor=16, mode='bilinear', align_corners=False),
        ], 1)

        empty_prediction = self.empty_predict(ipool)

        f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)

        return logit, empty_prediction, deep_layers
