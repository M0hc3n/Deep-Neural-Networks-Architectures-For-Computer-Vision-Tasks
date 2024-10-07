from core.config import device
from core.loggging import logger

from PIL import Image
import numpy as np

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

import torch

class DatasetTransformer(Dataset):
    cropping_box = None
    default_size = (64, 64)

    images_path = []

    training_images = None

    def __init__(self, images_path, cropping_box=(30, 55, 150, 175)):
        self.cropping_box = cropping_box
        self.images_path = images_path

        self.training_images = [
            np.array(Image.open(path).crop(self.cropping_box).resize(self.default_size))
            for path in self.images_path
        ]

        self.normalize_images()

        logger.info("Train Images collected: {}".format(len(self.training_images)))

    def normalize_images(self):
        for i in range(len(self.training_images)):
            self.training_images[i] = (
                self.training_images[i] - self.training_images[i].min()
            ) / (255 - self.training_images[i].min())

        self.training_images = np.array(self.training_images)

    def plotter(self):
        plt.figure(figsize=(10, 10))
        fig, ax = plt.subplots(2, 5)
        fig.suptitle("Real Images")
        idx = 8

        for i in range(2):
            for j in range(5):
                ax[i, j].imshow(self.training_images[idx].reshape(64, 64, 3))
                idx += 6
            
        plt.tight_layout()
        plt.show()

    def __len__(self):
        return len(self.training_images)

    def __getitem__(self, idx):
        image = self.training_images[idx]
        # Convert the image to a tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert HWC to CHW
        
        return image, 0
