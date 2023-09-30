import torch
import os
import os.path as op

from conv_block import *
from torch import nn
from YOLO_Layer import YoloLayer

import config.config as cfg
import utils


class DarkNet(nn.Module):
    def __init__(self, anchors, num_classes, num_blocks, img_dim):
        super().__init__()
        self.conv1 = BasicConv(3, 32, 3, stride=1, padding=1)
        self.res_block_1 = self.make_residual_block(64, num_blocks[0])
        self.res_block_2 = self.make_residual_block(128, num_blocks[1])
        self.res_block_3 = self.make_residual_block(256, num_blocks[2])
        self.res_block_4 = self.make_residual_block(512, num_blocks[3])
        self.res_block_5 = self.make_residual_block(1024, num_blocks[4])

        self.topdown_1 = Top_down(1024, 512)
        self.topdown_2 = Top_down(768, 256)
        self.topdown_3 = Top_down(384, 128)

        self.lateral_1 = BasicConv(512, 256, 1, stride=1, padding=0)
        self.lateral_2 = BasicConv(256, 128, 1, stride=1, padding=0)

        self.yolo_1 = YoloLayer(512, anchors[2], num_classes, img_dim)
        self.yolo_2 = YoloLayer(256, anchors[1], num_classes, img_dim)
        self.yolo_3 = YoloLayer(128, anchors[0], num_classes, img_dim)

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv1(x)
        c1 = self.res_block_1(x)
        c2 = self.res_block_2(c1)
        c3 = self.res_block_3(c2)
        c4 = self.res_block_4(c3)
        c5 = self.res_block_5(c4)

        p5 = self.topdown_1(c5)
        p4 = self.topdown_2(torch.cat((self.upsample(p5), self.lateral_1(c4)), 1))
        p3 = self.topdown_3(torch.cat((self.upsample(p4), self.lateral_2(c3)), 1))

        yolo_1 = self.yolo_1(p5)
        yolo_2 = self.yolo_2(p4)
        yolo_3 = self.yolo_3(p3)

        return torch.cat((yolo_1, yolo_2, yolo_3), 1), [yolo_1, yolo_2, yolo_3]

    def make_residual_block(self, input_channels, num_block):
        blocks = [BasicConv(input_channels // 2, input_channels, 3, stride=2, padding=1)]

        for i in range(num_block):
            blocks.append(ResidualBlock(input_channels))

        return nn.Sequential(*blocks)


def main():
    data_file = op.join(os.pardir, cfg.KITTI_JSON)
    data_config = utils.load_config(data_file)
    num_class = len(data_config["category2id"])
    num_blocks = [1, 2, 8, 8, 4]
    img_dims = data_config["image_input_size"]
    print(num_class, num_blocks, img_dims)
    anchors = [[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)], [(116, 90), (156, 198), (373, 326)]]
    x = torch.randn(1, 3, 416, 416)
    with torch.no_grad():
        model = DarkNet(anchors, num_class, num_blocks, img_dims)
        output_cat, output = model(x)
        print(output_cat.size())
        print(output[0].size(), output[1].size(), output[2].size())


if __name__ == "__main__":
    main()