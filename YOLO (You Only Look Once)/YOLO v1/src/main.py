from preparation.extract import VOCDatasetExtractor 
from preparation.transform import Compose
from modeling.model import Yolo
from modeling.train import ModelTrainer
from modeling.loss import Loss
# from modeling.train import ModelTrainer

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torch.optim as optim

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])

train_dataset = VOCDatasetExtractor(
    csv_file="./data/loader/train.csv", 
    image_dir="./data/loader/data/images", 
    label_dir="./data/loader/data/labels", 
    transform=transform
)

test_dataset = VOCDatasetExtractor(
    csv_file="./data/loader/test.csv", 
    image_dir="./data/loader/VOCdevkit/VOC2007/JPEGImages", 
    label_dir="./data/loader/VOCdevkit/VOC2007/labels", 
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=16,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=16,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)


model = Yolo(split_size=7, num_boxes=2, num_classes=20)
criterion = Loss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

model_trainer = ModelTrainer(
    training_data_loader=train_loader, 
    testing_data_loader=test_loader, 
    model=model, 
    criterion=criterion, 
    optimizer=optimizer, 
    epochs=20
)

model_trainer.train_model()
