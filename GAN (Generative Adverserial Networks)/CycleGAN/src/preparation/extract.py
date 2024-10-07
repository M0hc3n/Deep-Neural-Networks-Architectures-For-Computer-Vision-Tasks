from torchvision import transforms

from torch.utils.data import Dataset

import glob

import os

from PIL import Image

import random

class GetDataset(Dataset):
    def __init__(self, input_dir, transforms_=None, unaligned=False, mode="train"):
        super(GetDataset, self).__init__()

        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(input_dir, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(input_dir, "%sB" % mode) + "/*.*"))

    def convert_image(self, image):
        rgb_image = Image.new("RGB", image.size)

        return rgb_image.paste(image)

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        if self.is_gray_scale(image_A):
            image_A = self.convert_image(image_A)
        if self.is_gray_scale(image_B):
            image_B = self.convert_image(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return {"A": item_A, "B": item_B}

    def is_gray_scale(self, image):
        return image.mode != "RGB"

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
