from preparation.extract import ExtractDataset 
from preparation.transform import DatasetTransformer
from torch.utils.data import DataLoader
from modeling.generator_model import Generator
# from modeling.train import ModelTrainer

import torch

CelebA = ExtractDataset(input_dir="./data/raw/archive/img_align_celeba/img_align_celeba")

train_data = DatasetTransformer(images_path=CelebA.all_images)

generator = Generator()

noise = torch.randn(1, 100)

generator.eval()

with torch.no_grad():  # Disable gradient calculation for inference
    gen_image = generator(noise)

    print(gen_image)
   
# train_loaded_data = DataLoader(train_data, batch_size=32, shuffle=True)
# test_loaded_data = DataLoader(test_data, batch_size=32, shuffle=True)

# model_obj = Model()
# model, criterion, optimizer = model_obj.create_model()

# model_trainer = ModelTrainer(train_loaded_data,test_loaded_data , model, criterion, optimizer, epochs=10)
# model_trainer.train_model()

# model_trainer.plot_trainning_report()
# model_trainer.plot_model_weights()
