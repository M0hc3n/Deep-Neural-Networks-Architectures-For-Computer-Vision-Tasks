import torch
import torch.nn as nn
import torch.optim as optim

from core.config import device


class Generator(nn.Module):
    def __init__(self, noise_shape=100, image_channel=1, hidden_dim=64):
        super(Generator, self).__init__()

        # notice how we decrease the dimension to match the output one
        self.model = nn.Sequential(
            self.get_generator_block(
                input_channels=noise_shape, output_channels=hidden_dim * 4
            ),
            self.get_generator_block(
                input_channels=hidden_dim * 4,
                output_channels=hidden_dim * 2,
                kernel_size=4,
                stride=1,
            ),
            self.get_generator_block(
                input_channels=hidden_dim * 2, output_channels=hidden_dim
            ),
            self.get_generator_block(
                input_channels=hidden_dim,
                output_channels=image_channel,
                kernel_size=4,
                final_layer=True,
            ),
        )

    def get_generator_block(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=2,
        final_layer=False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride
                ),
                nn.Tanh(),
            )

    def forward(self, image):
        pred = self.model(image)
        
        return pred.view(len(pred), -1)
    
    

    # def create_model(self, learning_rate: float = 0.001):
    #     model = Generator().to(device)
    #     criterion = nn.BCELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    #     return model, criterion, optimizer

    # def load_model_from_checkpoint(
    #     self, checkpoint_path: str, input_shape: tuple, learning_rate: float = 0.001
    # ):
    #     model, criterion, optimizer = self.create_model(input_shape, learning_rate)
    #     model.load_state_dict(torch.load(checkpoint_path))
    #     model.eval()
    #     return model
