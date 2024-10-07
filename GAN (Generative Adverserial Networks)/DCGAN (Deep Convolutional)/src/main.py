from preparation.extract import DatasetExtractor
from preparation.transform import DatasetTransformer
from torch.utils.data import DataLoader

from modeling.gan import GAN
from modeling.train import ModelTrainer

CelebA = DatasetExtractor(
    input_dir="./data/raw/archive/img_align_celeba/img_align_celeba"
)

train_data = DatasetTransformer(images_path=CelebA.all_images)

train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model_obj = GAN()
model, criterion = model_obj.create_model()

noise_shape = 100
model_trainer = ModelTrainer(
    training_data_loader=train_data_loader,
    model=model,
    num_sample=len(train_data_loader),
    noise_shape=noise_shape,
    criterion=criterion,
    epochs=1,
)
model_trainer.train_model(generate_example=False)
model_trainer.generate_example()
model_trainer.plot_trainning_report()
print(model_trainer.loss_from_discriminator_model)
print(model_trainer.loss_from_generator_model)

