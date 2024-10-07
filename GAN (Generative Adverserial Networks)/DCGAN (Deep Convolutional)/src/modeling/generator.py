import torch
import torch.nn as nn
import torch.optim as optim

from core.config import device

class Generator(nn.Module):
    noise_shape = None

    def __init__(self, noise_shape=100):
        super(Generator, self).__init__()

        self.noise_shape = noise_shape

        self.model = nn.Sequential(
            # Input: noise_shape -> (4*4*512)
            nn.Linear(self.noise_shape, 4 * 4 * 512),
            nn.BatchNorm1d(4 * 4 * 512),
            nn.ReLU(inplace=True),  # or LeakyReLU if preferred

            # Reshape to (4, 4, 512)
            nn.Unflatten(1, (512, 4, 4)),
            
            # ConvTranspose2d layers
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Use Tanh to scale outputs to the range [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

    def create_model(self, learning_rate: float = 0.001):
        model = Generator().to(device)
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
