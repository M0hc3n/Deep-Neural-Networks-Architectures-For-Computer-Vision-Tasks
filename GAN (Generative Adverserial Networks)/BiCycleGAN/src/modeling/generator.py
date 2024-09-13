from torch import nn, cat, optim

from core.config import device

from modeling.unet import UnetDown, UnetUp

from core.hyperparameters import hp


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        channels, self.h, self.w = img_shape

        self.fc = nn.Linear(latent_dim, self.h * self.w)

        self.d1 = UnetDown(channels + 1, 64, normalize=False)
        self.d2 = UnetDown(64, 128)
        self.d3 = UnetDown(128, 256)
        self.d4 = UnetDown(256, 512)
        self.d5 = UnetDown(512, 512)
        self.d6 = UnetDown(512, 512)
        self.d6 = UnetDown(512, 512, normalize=False)

        self.u1 = UnetUp(512, 512)
        self.u2 = UnetUp(1024, 512)
        self.u3 = UnetUp(1024, 512)
        self.u4 = UnetUp(1024, 256)
        self.u5 = UnetUp(512, 128)
        self.u6 = UnetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, z):
        z = self.fc(z).view(z.size(0), 1, self.h, self.w)

        d1 = self.d1(cat((x, z), 1))
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)

        u1 = self.u1(d7, d6)
        u2 = self.u2(u1, d5)
        u3 = self.u3(u2, d4)
        u4 = self.u4(u3, d3)
        u5 = self.u5(u4, d2)
        u6 = self.u6(u5, d1)

        return self.final(u6)

    def create_model(self):
        model = Generator(self.latent_dim, self.img_shape).to(device)

        # initializing the optimizer of generator
        optim_G = optim.Adam(
            self.model.parameters(),
            lr=hp.lr,
            betas=(hp.b1, hp.b2),
        )

        return model, optim_G
