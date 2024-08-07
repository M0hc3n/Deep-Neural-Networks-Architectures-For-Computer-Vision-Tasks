from core.loggging import logger

import torchvision
class ExtractDataset:
    input_dir = ""
    train_data = None
    test_data = None

    train_classes = None
    train_targets = None

    test_classes = None
    test_targets = None

    def __init__(self, input_dir=""):
        self.input_dir = input_dir

        data_base = torchvision.datasets.FashionMNIST(
            f"{self.input_dir}/train", download=True, train=True
        )

        self.train_data = data_base.data
        self.train_targets = data_base.targets
        self.train_classes = data_base.classes

        test_base = torchvision.datasets.FashionMNIST(
            f"{self.input_dir}/test", download=True, train=False
        )

        self.test_data = test_base.data
        self.test_targets = test_base.targets
        self.test_classes = test_base.classes

        logger.info('Train collected: {}'.format(self.train_data.size()))
        logger.info('Test collected: {}'.format(self.test_data.size()))


