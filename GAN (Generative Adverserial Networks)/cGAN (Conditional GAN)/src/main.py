from preparation.extract import GetDataset

from modeling.gan import GAN
from modeling.train import ModelTrainer

batch_size = 128
noise_shape = 64

train_data_loader = GetDataset(
    input_dir="./data/raw/archive/fashion_mnist/fashion_mnist",
    batch_size=batch_size,
)

model_obj = GAN(
    noise_shape=noise_shape,
    data_shape=train_data_loader.shape,
    num_classes=train_data_loader.num_classes,
)
model, criterion = model_obj.create_model()

model_trainer = ModelTrainer(
    model=model,
    criterion=criterion,
    training_data_loader=train_data_loader.data_loader,
    noise_shape=noise_shape,
    images_shape=train_data_loader.shape,
    epochs=20,
)

model_trainer.train_model()
