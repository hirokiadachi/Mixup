import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################
#  Residual network (ResNet)
################################################
def conv3x3(in_features, out_features, stride=1):
    return nn.Conv2d(in_features, out_features, 
                     kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_features, out_features, stride=1, down=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_features, out_features, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_features)
        
        self.conv2 = conv3x3(out_features, out_features)
        self.bn2 = nn.BatchNorm2d(out_features)
        
        self.act = nn.ReLU(inplace=True)
        self.down = down
        
    def forward(self, x):
        res = x
        
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        
        h = self.conv2(h)
        h = self.bn2(h)
        
        if self.down is not None:
            res = self.down(x)
        h += res
        return self.act(h)
    
class ResNet(nn.Module):
    def __init__(self, in_ch, depth, classes=10):
        super(ResNet, self).__init__()
        n = (depth - 2) // 6
        block = BasicBlock
        
        self.in_ch = 16
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self.make_layer(block, 16, n, stride=2, down_size=True)
        self.layer2 = self.make_layer(block, 32, n, stride=2, down_size=True)
        self.layer3 = self.make_layer(block, 64, n, stride=2, down_size=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, out_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def make_layer(self, block, ch, blocks, stride=1, down_size=None):
        downsamle = None
        if stride != 1 or self.in_ch != ch * block.expansion:
            downsamle = nn.Sequential(
                nn.Conv2d(self.in_ch, ch * block.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch*block.expansion))
            
        layers = []
        layers.append(block(self.in_ch, ch, stride, downsamle))
        
        if down_size:
            self.in_ch = ch * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.in_ch, ch))
            return nn.Sequential(*layers)
        else:
            in_ch = ch * block.expansion
            for i in range(1, blocks):
                layers.append(block(in_ch, ch))
            return nn.Sequential(*layers)
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)    # 32x32

        h = self.layer1(h)  # 32x32
        h = self.layer2(h)  # 16x16
        h = self.layer3(h)
        h = self.avgpool(h)
        per_out = self.fc(torch.flatten(h, start_dim=1))
        return per_out, attn_out, attn_map
    
    
################################################
#  Simple CNN (Poor classifier)
################################################
class CNN(nn.Module):
    def __init__(self, in_ch=3, n_cls=10, img_size=32):
        super(CNN, self).__init__()
        n_convs = int(np.log2(img_size)) - 1
            
        layers = []
        out_ch = 16
        for _ in range(n_convs):
            layers.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch
            out_ch = out_ch * 2
        
        self.hidden_layers = nn.Sequential(*layers)
        fc_in = 2 * 2 * in_ch
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_cls))
        
    def forward(self, x):
        h = self.hidden_layers(x)
        h_flat = h.flatten(start_dim=1)
        out = self.classifier(h_flat)
        return out