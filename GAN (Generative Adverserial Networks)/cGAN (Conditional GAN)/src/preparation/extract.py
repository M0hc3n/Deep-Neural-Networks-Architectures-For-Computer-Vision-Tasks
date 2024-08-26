from core.loggging import logger

from torch.utils.data import DataLoader

from torchvision import transforms, datasets


class GetDataset:
    input_dir = ""

    def __init__(self, input_dir, batch_size):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        data_base = datasets.FashionMNIST(
            f"{input_dir}/train", download=True, transform=transform
        )

        self.data_loader = DataLoader(data_base, batch_size=batch_size, shuffle=True)

        self.num_classes = len(data_base.classes)

        for images, _ in self.data_loader:
            self.shape = images.shape[1:]
            break

        logger.info("Train collected: {}".format(len(self.data_loader)))
