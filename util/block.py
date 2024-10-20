from timm.models.vision_transformer import Block
import torch.nn as nn
import torch
import torch.nn.functional as F


class PAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(PAM, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Feature map transformation
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels // self.reduction_ratio, kernel_size=(1,1), stride=(1,1), padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels // self.reduction_ratio, out_channels=self.in_channels, kernel_size=(1,1), stride=(1,1), padding=0)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Global average pooling
        z = self.global_pool(x).view(batch_size, channels, 1, 1)

        # Feature map transformation
        z = self.conv1(z)
        z = F.relu(z, inplace=True)
        z = self.conv2(z)

        # Attention
        y = x.view(batch_size, channels, height * width)
        y = y.transpose(1, 2)
        y = torch.bmm(y, z)
        y = self.sigmoid(y)
        y = y.transpose(1, 2)
        y = y.view(batch_size, channels, height, width)

        # Output
        out = x * y

        return out


class block(Block):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dim = args[0]
        self.PAM = PAM(self.dim)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.PAM(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x