import torch.nn as nn
import torch.optim as optim

from core.config import device


class Discriminator(nn.Module):
    def __init__(self, image_channel=1, hidden_dim=64):
        super(Discriminator, self).__init__()

        self.image_channel = image_channel
        self.hidden_dim = hidden_dim

        # notice how we decrease the dimension to match the output one
        self.model = nn.Sequential(
            self.get_discriminator_block(
                input_channels=image_channel, output_channels=hidden_dim
            ),
            self.get_discriminator_block(
                input_channels=hidden_dim,
                output_channels=hidden_dim * 2,
            ),
            self.get_discriminator_block(
                input_channels=hidden_dim * 2,
                output_channels=1,
                final_layer=True,
            ),
        )

    def get_discriminator_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        final_layer=False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        disc_pred = self.model(image)

        return disc_pred.view(len(disc_pred), -1)

    def create_model(self, disc_input_dim, learning_rate: float = 0.001):
        model = Discriminator(disc_input_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        return model, criterion, optimizer

    # def load_model_from_checkpoint(
    #     self, checkpoint_path: str, input_shape: tuple, learning_rate: float = 0.001
    # ):
    #     model, criterion, optimizer = self.create_model(input_shape, learning_rate)
    #     model.load_state_dict(torch.load(checkpoint_path))
    #     model.eval()
    #     return model
