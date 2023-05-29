import os
import os.path as op

import numpy as np
import torch.nn.utils.rnn
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class ImageTransform:
    def __init__(self, image_size, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class BatchPadder:
    def __call__(self, sample):
        self.image = [sample['image'] for sample in sample]
        self.label = [sample['label'] for sample in sample]
        padded_input = torch.nn.utils.rnn.pad_sequence(self.image, batch_first=True)
        padded_label = self.get_numpy_from_nonfixed_2d_array(self.label)
        torch_label = torch.tensor(padded_label)
        return {"image": padded_input.contiguous(),
                "label": torch_label.contiguous()}

    def get_numpy_from_nonfixed_2d_array(self, input_label_list):
        padding_list = [-1, 0, 0, 0, 0]
        rows = []
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
    def __init__(self, file_list,  transform=None):
        self.file_list = glob(op.join(file_list, "image_2", "*.png"))
        self.file_list.sort()
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image = self.file_list[index]
        label = self.get_label(self.file_list[index])
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        sample = {"image": image, "label": label}
        return sample

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
        x1, y1, x2, y2, depth = int(round(float(label[4]))), int(round(float(label[5]))), \
            int(round(float(label[6]))), int(round(float(label[7]))), int(round(float(label[-2])))
        label_info = [category_id, (x2 + x1) / 2, (y2 + y1) / 2, (x2 - x1), (y2 - y1)]
        return label_info

    def convert_category2id(self, category):
        category_id = 1
        return category_id


def main(data_path):
    mean = (0.5, )
    std = (0.3, )
    image_size = (256, 832)
    training_data_list = op.join(data_path, "training")
    test_data_list = op.join(data_path, "test")
    training_data = KiTTiDataset(training_data_list, transform=ImageTransform(image_size, mean, std))
    testing_data = KiTTiDataset(test_data_list, transform=ImageTransform(image_size, mean, std))

    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True, collate_fn=BatchPadder())
    # test_dataloader = DataLoader(testing_data, batch_size=4, shuffle=True, collate_fn=BatchPadder())

    for img, lbl in train_dataloader:
        print(img)
        print(lbl)


if __name__ == "__main__":
    data_path = "/mnt/intHDD/kitti/"
    main(data_path)
