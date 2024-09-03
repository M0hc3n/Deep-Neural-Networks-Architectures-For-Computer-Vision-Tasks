class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    epoch=0,
    n_epochs=200,
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=4,
    lr=0.0002,
    decay_start_epoch=100,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    img_size=128,
    channels=3,
    n_critic=5,
    sample_interval=100,
    num_residual_blocks=19,
    lambda_cyc=10.0,
    lambda_id=5.0,
)
