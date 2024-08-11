from core.config import device
from core.loggging import logger

from PIL import Image
import numpy as np

from torch.utils.data import Dataset


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
