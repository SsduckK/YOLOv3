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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stride = 0
        self.grid_x = 0
        self.grid_y = 0
        self.scaled_anchors = 0
        self.anchor_w = 0
        self.anchor_h = 0

        self.conv = nn.Sequential(
            BasicConv(channels, channels * 2, 3, stride=1, padding=1),
            nn.Conv2d(channels * 2, (self.num_classes + 5) * self.num_anchors, 1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.conv(x)
        batch_size = x.size(0)
        grid_size = x.size(2)
        print(x.shape)
        prediction = x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2)
        prediction = prediction.contiguous()

        obj_score = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        pred_bboxes = self.transform_outputs(prediction)
        output = torch.cat((pred_bboxes.view(batch_size, -1, 4), obj_score.view(batch_size, -1, 1),
                            pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        self.stride = self.img_dim / self.grid_size

        self.grid_x = torch.arange(grid_size, device=self.device).repeat(1, 1, grid_size, 1).type(torch.float32)
        self.grid_y = (torch.arange(grid_size, device=self.device).repeat(1, 1, grid_size, 1).
                       transpose(3, 2).type(torch.float32))

        scaled_anchors = [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]
        self.scaled_anchors = torch.tensor(scaled_anchors, device=self.device)

        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def transform_outputs(self, prediction):
        device = prediction.device
        x = torch.sigmoid(prediction[..., 0]).cuda()
        y = torch.sigmoid(prediction[..., 1]).cuda()
        w = prediction[..., 2].cuda()
        h = prediction[..., 3].cuda()

        pred_bboxes = torch.zeros_like(prediction[..., :4].to(device))
        pred_bboxes[..., 0] = x.data + self.grid_x
        pred_bboxes[..., 1] = y.data + self.grid_y
        pred_bboxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_bboxes[..., 3] = torch.exp(h.data) * self.anchor_h

        return pred_bboxes * self.stride


