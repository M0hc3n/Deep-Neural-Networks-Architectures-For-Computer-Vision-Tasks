import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from modeling.gan import GAN
from core.config import report_dir, Tensor
from core.hyperparameters import hp

import time
import datetime
import sys

from torch.autograd import Variable

from torchvision.utils import save_image
from IPython.display import clear_output


class ModelTrainer:
    training_data_loader = None

    model = None

    loss_from_discriminator_model = []
    loss_from_generator_model = []

    epochs = 5

    def __init__(
        self,
        training_data_loader,
        validation_data_loader,
        model: GAN,
        input_shape,
        epochs,
    ):
        self.training_data_loader = training_data_loader
        self.val_dataloader = validation_data_loader

        self.model = model

        self.epochs = epochs

        self.input_shape = input_shape

    def _train_conditional_vae(self, real_A, real_B, valid):
        mu, logvar, reparamterized_out = self.model.enc(real_B)

        fake_B = self.model.gen(real_A, reparamterized_out)

        pixel_wise_l1_loss_vae = self.model.mae_loss(real_B, fake_B)

        loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - logvar - 1)

        loss_vae_GAN = self.model.D_VAE.compute_loss(fake_B, valid)

        return fake_B, pixel_wise_l1_loss_vae, loss_kl, loss_vae_GAN

    def _sample_from_normal(self, size):
        return Variable(Tensor(np.random.normal(0, 1, size)))

    def _train_conditional_lr(self, real_A, valid):
        sampled_z = self._sample_from_normal(size=(real_A.size(0), hp.latent_dim))

        _fake_B = self.model.gen(real_A, sampled_z)

        loss_clr_GAN = self.model.D_LR.compute_loss(_fake_B, valid)

        return sampled_z, _fake_B, loss_clr_GAN

    def train_model(
        self, lambda_pix, lambda_kl, lambda_latent, start_epoch=0, sample_interval=100
    ):
        start = time.time()

        valid = 1
        fake = 0

        if start_epoch >= self.epochs:
            raise Exception("Starting Epoch exceeds Max Epochs")

        for epoch in range(start_epoch, self.epochs):
            print(f"Currently training on Epoch {epoch+1}")

            for i, batch in enumerate(self.training_data_loader):
                real_A = Variable(batch["A"].type(Tensor))
                real_B = Variable(batch["B"].type(Tensor))

                self.model.optim_G.zero_grad()
                self.model.optim_E.zero_grad()

                # we train the first wheel of the bicyle: ConditionalVAE
                fake_B, pixel_wise_l1_loss_vae, loss_kl, loss_vae_GAN = (
                    self._train_conditional_vae(real_A, real_B, valid)
                )

                # we train the second wheel of the bicycle: Conditional LR
                sampled_z, _fake_B, loss_clr_GAN = self._train_conditional_lr(
                    real_A, valid
                )

                # calculate generator + encoder loss:
                gen_enc_loss = (
                    loss_vae_GAN
                    + loss_clr_GAN
                    + lambda_pix * pixel_wise_l1_loss_vae
                    + lambda_kl * loss_kl
                )

                gen_enc_loss.backward(retain_graph=True)
                self.model.optim_E.step()

                # calculate generator only loss:
                _mu, _, _ = self.model.enc(_fake_B)
                loss_latent = lambda_latent * self.model.mae_loss(_mu, sampled_z)

                loss_latent.backward()
                self.model.optim_G.step()

                # train discriminator: (cVAE)
                self.model.optim_D_VAE.zero_grad()

                loss_D_VAE = self.model.D_VAE.compute_loss(
                    real_B, fake
                ) + self.model.D_VAE.compute_loss(fake_B.detach(), valid)

                loss_D_VAE.backward()
                self.model.optim_D_VAE.step()

                # train discriminator: (cLR)
                self.model.optim_D_LR.zero_grad()

                loss_D_LR = self.model.D_LR.compute_loss(
                    real_B, fake
                ) + self.model.D_LR.compute_loss(_fake_B.detach(), valid)

                loss_D_LR.backward()
                self.model.optim_D_LR.step()

                # Determine approximate time left
                batches_done = epoch * len(self.training_data_loader) + i
                batches_left = (
                    hp.n_epochs * len(self.training_data_loader) - batches_done
                )
                time_left = datetime.timedelta(
                    seconds=batches_left * (time.time() - start)
                )
                start = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent: %f] ETA: %s"
                    % (
                        epoch,
                        hp.n_epochs,
                        i,
                        len(self.training_data_loader),
                        loss_D_VAE.item(),
                        loss_D_LR.item(),
                        gen_enc_loss.item(),
                        pixel_wise_l1_loss_vae.item(),
                        loss_kl.item(),
                        loss_latent.item(),
                        time_left,
                    )
                )

                # If at sample interval save image
                if batches_done % hp.sample_interval == 0:
                    clear_output()
                    self.visualise_output(self.sample_images(batches_done), 30, 10)

    def plot_output(path, x, y):
        img = mpimg.imread(path)
        plt.figure(figsize=(x, y))
        plt.imshow(img)
        plt.show()

    def sample_images(self, batches_done):
        """From the validation set this method will create images and
        save those Generated samples in a path"""
        self.model.gen.eval()
        imgs = next(iter(self.val_dataloader))
        # next() will supply each subsequent element from the iterable
        # So in this case each subsequent set of images from val_dataloader
        img_samples = None
        # For below line to work, I need to create a folder named 'maps' in the root_path
        path = "/content/%s/%s.png" % ("maps", batches_done)
        for img_A, img_B in zip(imgs["A"], imgs["B"]):
            # Repeat input image by number of desired columns
            real_A = img_A.view(1, *img_A.shape).repeat(hp.latent_dim, 1, 1, 1)
            real_A = Variable(real_A.type(Tensor))
            # Sample latent representations
            sampled_z = Variable(
                Tensor(np.random.normal(0, 1, (hp.latent_dim, hp.latent_dim)))
            )
            # Generate samples
            fake_B = self.model.gen(real_A, sampled_z)
            # Concatenate samples horizontally
            fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
            img_sample = torch.cat((img_A, fake_B), -1)
            img_sample = img_sample.view(1, *img_sample.shape)
            # Concatenate with previous samples vertically
            img_samples = (
                img_sample
                if img_samples is None
                else torch.cat((img_samples, img_sample), -2)
            )
        save_image(img_samples, path, nrow=8, normalize=True)
        self.model.gen.train()
        return path
