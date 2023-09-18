import os
import os.path as op

import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

from PIL import Image
from glob import glob


import torch.nn.utils.rnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


class ImageTransform:
    def __init__(self, image_size, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img, label):
        return self.data_transform(img), label


class BatchPadder:
    def __call__(self, sample):
        image = [sample['image'] for sample in sample]
        label = [sample['label'] for sample in sample]
        padded_input = torch.nn.utils.rnn.pad_sequence(image, batch_first=True)
        padded_label = self.get_numpy_from_nonfixed_2d_array(label)
        torch_label = torch.tensor(padded_label)
        return {"image": padded_input.contiguous(),
                "label": torch_label.contiguous()}

    def get_numpy_from_nonfixed_2d_array(self, input_label_list):
        padding_list = [-1, 0, 0, 0, 0]
        size_of_list = []
        for i in input_label_list:
            size_of_list.append(len(i))
        max_length = max(size_of_list)
        for line in input_label_list:
            if len(line) < max_length:
                for pad_array in range(max_length - len(line)):
                    line.append(padding_list)
        return input_label_list


class KiTTiDataset(Dataset):
    def __init__(self, file_list, config_path, transform=None):
        self.file_list = glob(op.join(file_list, "image_2", "*.png"))
        self.file_list.sort()
        self.config_data = self.load_config(op.join(config_path, "kitti_config.json"))
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image = self.file_list[index]
        label = self.get_label(self.file_list[index])
        image = Image.open(image)
        if self.transform:
            image, label = self.transform(image, label)
        sample = {"image": image, "label": label}
        return sample

    def load_config(self, config_path):
        config_file = op.join(config_path)
        with open(config_file) as f:
            config_data = json.load(f)
        return config_data

    def get_label(self, file_list):
        label_file = file_list.replace("image_2", "label_2").replace(".png", ".txt")
        label_info = []
        with open(label_file, 'r') as f:
            label_lines = f.readlines()
            for label_line in label_lines:
                label_line = label_line.strip().split(" ")
                label_info_per_line = self.get_bbox(label_line)
                label_info.append(label_info_per_line)
        return label_info

    def get_bbox(self, label):
        category = label[0]
        category_id = self.convert_category2id(category)
        x, y, w, h, depth = int(round(float(label[4]))), int(round(float(label[5]))), \
            int(round(float(label[6]))), int(round(float(label[7]))), int(round(float(label[-2])))
        label_info = [category_id, x, y, w, h]
        return label_info

    def convert_category2id(self, category):
        category2id = self.config_data["category2id"]
        category_id = category2id[category]
        return category_id


def image_show(dataloader):
    data = next(iter(dataloader))
    train_features, train_labels = data["image"], data["label"]
    img = train_features[0].squeeze()
    label = train_labels[0]
    draw_line(img, label)


def draw_line(img, labels):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for lbl in labels:
        if lbl[0] != -1:
            img = cv2.rectangle(img, (int(lbl[1]), int(lbl[2])), (int(lbl[3]), int(lbl[4])), (255, 0, 0))
    cv2.imshow("image", img)
    cv2.waitKey()


def main(data_path, config_path):
    mean = (0.5, )
    std = (0.3, )
    image_size = (370, 1224)
    training_data_list = op.join(data_path, "training")
    test_data_list = op.join(data_path, "testing")
    training_data = KiTTiDataset(training_data_list,  config_path, transform=ImageTransform(image_size, mean, std))
    testing_data = KiTTiDataset(test_data_list, config_path, transform=ImageTransform(image_size, mean, std))
    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True, collate_fn=BatchPadder())

    image_show(train_dataloader)


if __name__ == "__main__":
    data_path = "/mnt/intHDD/kitti/"
    config_path = "/home/gorilla/lee_ws/YOLOv3/config"
    main(data_path, config_path)
