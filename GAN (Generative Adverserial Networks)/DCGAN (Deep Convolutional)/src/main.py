from preparation.extract import ExtractDataset
from preparation.transform import DatasetTransformer
from torch.utils.data import DataLoader

from modeling.gan import GAN
from modeling.train import ModelTrainer

CelebA = ExtractDataset(
    input_dir="./data/raw/archive/img_align_celeba/img_align_celeba"
)

train_data = DatasetTransformer(images_path=CelebA.all_images)

train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model_obj = GAN()
model, criterion = model_obj.create_model()

noise_shape = 100
model_trainer = ModelTrainer(
    train_data_loader,
    model,
    len(train_data_loader),
    noise_shape,
    criterion,
    epochs=50,
)
model_trainer.train_model()
model_trainer.generate_example()
# model_trainer.plot_trainning_report()
# model_trainer.plot_model_weights()
