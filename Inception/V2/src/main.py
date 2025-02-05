from preparation.extract import GetDataset

from modeling.model import InceptionV2
from core.config import device

from modeling.train import ModelTrainer

data = GetDataset(
    input_dir="data/raw",
    batch_size=32,
)

# model = InceptionV2(num_classes=data.num_classes).to(device)

# model_trainer = ModelTrainer(
#     model=model,
#     train_data=data.train_loader,
#     validation_data=data.valid_loader,
#     test_data=data.test_loader,
#     optimizer=model.get_optimizer(),
#     criterion=model.get_criterion(),
#     epochs=15,
# )
# model_trainer.fit()
# model_trainer.plot_trainning_report()

# model_trainer.test()
# model_trainer.plot_testing_report()
