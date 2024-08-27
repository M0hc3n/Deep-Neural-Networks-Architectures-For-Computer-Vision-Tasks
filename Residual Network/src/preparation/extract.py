from torchvision import transforms, datasets

from torch.utils.data import DataLoader, random_split

class GetDataset:
    input_dir = ""

    def __init__(self, input_dir, batch_size):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        train_valid_data = datasets.CIFAR10(
            f"{input_dir}/train", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            f"{input_dir}/test", train=False, download=True, transform=transform
        )

        train_dataset, valid_dataset = random_split(train_valid_data, (45000, 5000))

        print(
            "Image shape of a random sample image : {}".format(
                train_dataset[0][0].numpy().shape
            ),
            end="\n\n",
        )

        print("Training Set:   {} images".format(len(train_dataset)))
        print("Validation Set:   {} images".format(len(valid_dataset)))
        print("Test Set:       {} images".format(len(test_dataset)))

        batch_size = 32

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
