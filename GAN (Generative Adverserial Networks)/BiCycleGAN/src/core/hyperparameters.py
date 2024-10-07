class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    epoch=0,
    n_epochs=200,
    batch_size=8,
    dataset_train_mode="train",
    dataset_test_mode="val",
    lr=0.0002,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    img_size=128,
    channels=3,
    latent_dim=8,
    n_critic=5,
    sample_interval=400,
    lambda_pixel=10,
    lambda_latent=0.5,
    lambda_kl=0.01,
)
