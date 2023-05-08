# -*- coding: UTF-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(args.num_channels, 64, 3, padding="same", bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(64, 64, 3, padding="same", bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('pool1', nn.MaxPool2d((2, 2), (2, 2))),
            ('dropout1', nn.Dropout(0.25)),
            ('conv3', nn.Conv2d(64, 128, 3, padding="same", bias=False)),
            ('bn3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(128, 128, 3, padding="same", bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU()),
            ('pool2', nn.MaxPool2d((2, 2), (2, 2))),
            ('dropout2', nn.Dropout(0.25)),
            ('conv5', nn.Conv2d(128, 256, 3, padding="same", bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU()),
            ('conv6', nn.Conv2d(256, 256, 3, padding="same", bias=False)),
            ('bn6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU()),
            ('conv7', nn.Conv2d(256, 256, 3, padding="same", bias=False)),
            ('bn7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU()),
            ('pool3', nn.MaxPool2d((2, 2), (2, 2))),
            ('dropout3', nn.Dropout(0.25)),
            ('conv8', nn.Conv2d(256, 512, 3, padding="same", bias=False)),
            ('bn8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU()),
            ('conv9', nn.Conv2d(512, 512, 3, padding="same", bias=False)),
            ('bn9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU()),
            ('conv10', nn.Conv2d(512, 512, 3, padding="same", bias=False)),
            ('bn10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU()),
            ('pool4', nn.MaxPool2d((2, 2), (2, 2))),
            ('dropout4', nn.Dropout(0.25)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ]))
        self.fc = nn.Linear(512, args.num_classes)

    def forward(self, x):
        output = self.model(x)
        output = output.view(output.shape[0], -1)
        return self.fc(output)


class CNN4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(args.num_channels, 64, kernel_size=3)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d((2, 2))),
            ('conv2', nn.Conv2d(64, 128, kernel_size=3)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d((2, 2)))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2304 * 2, 512)),
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(512, args.num_classes)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = torch.flatten(x, 1)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, args):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, args.num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(args):
    return ResNet(ResidualBlock, args)


def get_model(args):
    if args.model == 'VGG16':
        return VGG16(args)
    elif args.model == 'CNN4':
        return CNN4(args)
    elif args.model == 'ResNet18':
        return ResNet18(args)
    else:
        exit("Unknown Model!")
