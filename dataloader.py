import os
import os.path as op
import cv2
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class KiTTiDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = glob(op.join(file_list, "image_2", "*.png"))
        self.file_list.sort()
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image = self.file_list[index]
        label = self.get_label(self.file_list[index])
        image = cv2.imread(image)
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_label(self, file_list):
        label_file = file_list.replace("image_2", "label_2").replace(".png", ".txt")
        categories = []
        bbox2ds = []
        with open(label_file, 'r') as f:
            label_lines = f.readlines()
            for label_line in label_lines:
                label_line = label_line.strip()
                label_line = label_line.split(" ")
                category, bbox2d = self.get_bbox(label_line)
                categories.append(category)
                bbox2ds.append(bbox2d)
        return [categories, bbox2ds]

    def get_bbox(self, label_info):
        category = label_info[0]
        x1, y1, x2, y2, depth = int(round(float(label_info[4]))), int(round(float(label_info[5]))), \
            int(round(float(label_info[6]))), int(round(float(label_info[7]))), int(round(float(label_info[-2])))
        bbox2d = [(x2 + x1) / 2, (y2 + y1) / 2, (x2 - x1), (y2 - y1)]
        return category, bbox2d


def main(data_path):
    training_data_list = op.join(data_path, "training")
    test_data = op.join(data_path, "test")
    training_data = KiTTiDataset(training_data_list)

    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
    for img, lbl in train_dataloader:
        print(img)
        print(lbl)


if __name__ == "__main__":
    data_path = "/mnt/intHDD/kitti/"
    main(data_path)
