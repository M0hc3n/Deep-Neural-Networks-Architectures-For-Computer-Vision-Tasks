from preparation.extract import GetDataset

from modeling.gan import GAN
from modeling.train import ModelTrainer

train_data_loader = GetDataset(
    input_dir="./data/raw/archive/fashion_mnist/fashion_mnist", batch_size=128
)

model_obj = GAN(random_shape=64, data_shape=(1, 28, 28), num_classes=10)
model, criterion = model_obj.create_model()

noise_shape = 100

model_trainer = ModelTrainer(
    training_data_loader=train_data_loader.data_loader,
    model=model,
    noise_shape=64,
    criterion=criterion,
    epochs=20,
    images_shape=(1,28,28),
)

model_trainer.train_model()
