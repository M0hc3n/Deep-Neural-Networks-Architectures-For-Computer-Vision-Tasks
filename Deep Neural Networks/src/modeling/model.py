import torch
import torch.nn as nn
import torch.optim as optim

from core.config import device
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.input_to_hidden_layer = nn.Linear(784, 1000)
        self.batch_norm = nn.BatchNorm1d(1000)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        fc0 = self.batch_norm(x)
        fc1 = self.hidden_layer_activation(fc0)
        fc2 = self.hidden_to_output_layer(fc1)
        return fc2, fc1

    def create_model(self, learning_rate: float = 0.001):
        model = Model().to(device)
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
