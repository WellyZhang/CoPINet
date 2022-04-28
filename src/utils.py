# -*- coding: utf-8 -*-

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def imresize(img, size):
    return np.array(Image.fromarray(img).resize(size))


class dataset(Dataset):

    def __init__(self, dataset_path, dataset_type, img_size, test=False):
        self.dataset_path = dataset_path
        if not test:
            self.file_names = [
                f for f in glob.glob(
                    os.path.join(self.dataset_path, "*", "*.npz"))
                if dataset_type in f
            ]
        else:
            self.file_names = [
                f for f in glob.glob(os.path.join(self.dataset_path, "*.npz"))
                if dataset_type in f
            ]
        self.img_size = img_size

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]

        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160)
        target = data["target"]

        if self.img_size != 160:
            resize_image = []
            for idx in range(16):
                resize_image.append(
                    imresize(image[idx, :, :], (self.img_size, self.img_size)))
            image = np.stack(resize_image)

        image = torch.tensor(image, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.long)

        return image, target
