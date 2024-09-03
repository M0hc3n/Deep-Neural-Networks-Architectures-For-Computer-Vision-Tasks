from core.loggging import logger

from PIL import Image

from torchvision import transforms

from torch.utils.data import Dataset

from torchvision.utils import make_grid

from preparation.extract import GetDataset

from torch.utils.data import DataLoader

from core.hyperparameters import hp

from preparation.plotter import Plotter

from core.config import plot_dir


class DatasetLoader(Dataset):
    def __init__(
        self,
        input_path,
    ):
        transforms_ = [
            transforms.Resize((hp.img_size, hp.img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),  # this one is the mean
                (0.5, 0.5, 0.5),  # this one is the std
            ),
        ]

        self.train_dataloader = DataLoader(
            GetDataset(input_path, mode=hp.dataset_train_mode, transforms_=transforms_),
            batch_size=hp.batch_size,
            shuffle=True,
            num_workers=1,
        )

        self.val_dataloader = DataLoader(
            GetDataset(input_path, mode=hp.dataset_test_mode, transforms_=transforms_),
            batch_size=16,
            shuffle=True,
            num_workers=1,
        )

        logger.info("Train Images collected: {}".format(len(self.train_dataloader)))
        logger.info("Validation Images collected: {}".format(len(self.val_dataloader)))

    def show_example(self, pic_size=2):
        data_iter = iter(self.train_dataloader)
        images = next(data_iter)

        for i in range(len(images["A"])):
            Plotter().show_img(
                make_grid(images["A"][i], images["B"][i]),
                size=pic_size,
                figname=f"{plot_dir}/example {i}",
            )
