import torch
import torch.nn as nn
import torch.optim as optim

from core.config import device
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(28 * 28, 1000), nn.ReLU(), nn.Linear(1000, 10)
        ).to(device)

    def forward(self, x):
        return self.model(x)

    def create_model(self, learning_rate: float = 0.001):
        model = Model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        return model, criterion, optimizer

    def load_model_from_checkpoint(
        self, checkpoint_path: str, input_shape: tuple, learning_rate: float = 0.001
    ):
        model, criterion, optimizer = self.create_model(input_shape, learning_rate)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        return model
