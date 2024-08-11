from preparation.extract import ExtractDataset 
from preparation.transform import DatasetTransformer
from torch.utils.data import DataLoader
from modeling.model import Model
from modeling.train import ModelTrainer

MNISTFashion = ExtractDataset(input_dir="./data/raw")

train_data = DatasetTransformer(MNISTFashion.train_data, MNISTFashion.train_targets)
# test_data = DatasetTransformer(MNISTFashion.test_data, MNISTFashion.test_targets)

train_loaded_data = DataLoader(train_data, batch_size=32, shuffle=True)
# test_loaded_data = DataLoader(test_data, batch_size=32, shuffle=True)

model_obj = Model()
model, criterion, optimizer = model_obj.create_model()

# model_trainer = ModelTrainer(train_loaded_data,test_loaded_data , model, criterion, optimizer, epochs=10)
# model_trainer.train_model()

# model_trainer.plot_trainning_report()
# model_trainer.plot_model_weights()
