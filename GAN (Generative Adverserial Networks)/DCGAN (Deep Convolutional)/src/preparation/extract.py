import os

from core.loggging import logger

class ExtractDataset:
    input_dir = ""

    all_images = []

    def __init__(self, input_dir=""):
        self.input_dir = input_dir

        for path in os.listdir(self.input_dir):
            if '.jpg' in path:
                self.all_images.append(os.path.join(self.input_dir, path))

        # did it for optimization
        self.all_images = self.all_images[0:500] # take only first 500s 

        logger.info('Train collected: {}'.format(len(self.all_images)))

