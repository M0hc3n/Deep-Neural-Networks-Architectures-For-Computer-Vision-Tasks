from preparation.extract import ExtractDataset 
from preparation.transform import DatasetTransformer
from torch.utils.data import DataLoader
from modeling.model import Model
from modeling.train import ModelTrainer

MNISTFashion = ExtractDataset(input_dir="./data/raw")

train_data = DatasetTransformer(MNISTFashion.train_data, MNISTFashion.targets)

train_loaded_data = DataLoader(train_data, batch_size=32, shuffle=True)

model_obj = Model()
model, criterion, optimizer = model_obj.create_model()

model_trainer = ModelTrainer(train_loaded_data, model, criterion, optimizer)
model_trainer.train_model()

model_trainer.plot_trainning_report()