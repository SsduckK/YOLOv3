import torch

from conv_block import *
from torch import nn


class YoloLayer(nn.Module):
    def __init__(self, channels, anchors, num_classes, img_dim):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0

        self.conv = nn.Sequential(
            BasicConv(channels, channels*2, 3, stride=1, padding=1),
            nn.Conv2d(channels*2, 75, 1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.conv(x)

        batch_size = x.size(0)
        grid_size = x.size(2)
        device = x.device

        prediction = x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
        prediction = prediction(0, 1, 3, 4, 2)
        prediction = prediction.contigouous()

        obj_score = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

