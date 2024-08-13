import torch
import torch.nn as nn
import torch.optim as optim

from core.config import device

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input shape: (batch_size, 3, 64, 64) -> (batch_size, 64, 32, 32)
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            
            # (batch_size, 64, 32, 32) -> (batch_size, 64, 16, 16)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            
            # Flatten the tensor to (batch_size, num_features)
            nn.Flatten(),
            
            # Final dense layer: (batch_size, num_features) -> (batch_size, 1)
            nn.Linear(64 * 16 * 16, 1),
            nn.Sigmoid()  # Output activation function
        )
    

    def forward(self, x):
        return self.model(x)

    def create_model(self, learning_rate: float = 0.001):
        model = Discriminator().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        return model, criterion, optimizer

    def load_model_from_checkpoint(
        self, checkpoint_path: str, input_shape: tuple, learning_rate: float = 0.001
    ):
        model, criterion, optimizer = self.create_model(input_shape, learning_rate)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        return model
