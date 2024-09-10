from preparation.transform import DatasetLoader

from core.hyperparameters import hp

from modeling.gan import GAN
from modeling.train import ModelTrainer

if __name__ == "__main__":
    loaded_data = DatasetLoader(
        input_path="./data/raw/archive",
    )

    loaded_data.show_example()
    # input_shape = (hp.channels, hp.img_size, hp.img_size)

    # model = GAN(input_shape)

    # trainer = ModelTrainer(
    #     training_data_loader=loaded_data.train_dataloader,
    #     validation_data_loader=loaded_data.val_dataloader,
    #     model=model,
    #     input_shape=input_shape,
    #     epochs=300,
    # )

    # trainer.train_model(lambda_id=hp.lambda_id, lambda_cyc=hp.lambda_cyc)
