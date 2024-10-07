from torchvision import transforms

from torch.utils.data import Dataset

import glob

import os

from PIL import Image

import numpy as np


class GetDataset(Dataset):
    def __init__(self, input_dir, transforms_=None, unaligned=False, mode="train"):
        super(GetDataset, self).__init__()

        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(input_dir, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(
                sorted(glob.glob(os.path.join(input_dir, "test") + "/*.*"))
            )

    def convert_image(self, image):
        rgb_image = Image.new("RGB", image.size)

        return rgb_image.paste(image)

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])

        w, h = img.size

        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        # apply mirroring at 0.5 probability
        if np.random.random() < 0.5:
            # the use of ::-1 reverses the middle dimension (the width),
            # which results in the image getting flipped horizontally
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def is_gray_scale(self, image):
        return image.mode != "RGB"

    def __len__(self):
        return len(self.files)
