from core.loggging import logger

import torchvision
class ExtractDataset:
    input_dir = ""
    train_data = None

    classes = None
    targets = None

    def __init__(self, input_dir=""):
        self.input_dir = input_dir

        data_base = torchvision.datasets.FashionMNIST(
            f"{self.input_dir}/train", download=True, train=True
        )

        self.train_data = data_base.data
        self.targets = data_base.targets
        self.classes = data_base.classes

        logger.info('collected: {}'.format(self.train_data.size()))

