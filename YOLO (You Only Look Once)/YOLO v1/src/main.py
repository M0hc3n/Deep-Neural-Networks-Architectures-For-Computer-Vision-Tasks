from preparation.extract import VOCDatasetExtractor 
# from preparation.transform import DatasetTransformer
# from torch.utils.data import DataLoader
# from modeling.model import Model
# from modeling.train import ModelTrainer

import torchvision.transforms as transforms

from torch.utils.data import DataLoader


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for transform in self.transforms:
            img, bboxes = transform(img), bboxes
        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])

train_dataset = VOCDatasetExtractor(csv_file="./data/loader/train.csv", image_dir="./data/loader/data/images", label_dir="./data/loader/data/labels")
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=16,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

print(train_dataset)