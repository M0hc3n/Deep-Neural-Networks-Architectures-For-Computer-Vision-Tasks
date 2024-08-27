from preparation.extract import GetDataset

from modeling.model import ResNet
from modeling.utils.basic_conv_block import BasicConvBlock

from core.config import device

from modeling.train import ModelTrainer

data = GetDataset(input_dir="./data/raw")

model_obj = ResNet(block_type=BasicConvBlock, num_blocks=[9, 9, 9, 9]).to(device)
model, criterion, optimizer = model_obj.create_model()

model_trainer = ModelTrainer(
    train_loaded_data, test_loaded_data, model, criterion, optimizer, epochs=10
)
model_trainer.train_model()

model_trainer.plot_trainning_report()
model_trainer.plot_model_weights()
