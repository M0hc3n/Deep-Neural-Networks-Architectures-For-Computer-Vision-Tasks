import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from modeling.gan import GAN
from core.config import report_dir, Tensor

import time
import datetime

from torch.autograd import Variable

from torchvision.utils import save_image
from torchvision.utils import make_grid

from IPython.display import clear_output


class ModelTrainer:
    training_data_loader = None

    model, criterion = None, None

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

    def _calculate_identity_loss(self, real_A, real_B):
        # Recall the concept of identity loss (introduced in the paper)
        # it specifies that the generators need to be aware of domain A and B
        # and if fed an images from domain B (instead of the base A), it needs
        # return the same image, as a proof that it has nothing to do to it
        pred = self.model.gen_BA(real_A)

        print(pred.shape, real_A.shape)
        identity_loss_A = self.model.identity_criterion(
            pred, real_A
        )  # notice how we are passing real_A (instead of real_B) even though
        # the generator model is supposed to be fed real_B
        # this loss measures the generator's awareness of the differences between
        # the domains A and B
        identity_loss_B = self.model.identity_criterion(
            self.model.gen_AB(real_B), real_B
        )  # we do the same with gen_AB

        return (identity_loss_A + identity_loss_B) / 2

    def _calculate_GAN_loss(self, real_A, real_B, valid):
        # loss for GAN_AB
        fake_B = self.model.gen_AB(real_A)
        loss_GAN_AB = self.model.criterion_GAN(self.model.disc_B(fake_B), valid)

        # loss for GAN_BA
        fake_A = self.model.gen_BA(real_B)
        loss_GAN_BA = self.model.criterion_GAN(self.model.disc_A(fake_A), valid)

        return (loss_GAN_AB + loss_GAN_BA) / 2, fake_A, fake_B

    def _calculate_cycle_loss(self, real_A, real_B, fake_A, fake_B):
        # now, we compute the cycle consistency loss
        reconstructed_A = self.model.gen_BA(fake_B)

        cycle_loss_A = self.model.cycle_criterion(real_A, reconstructed_A)

        reconstructed_B = self.model.gen_AB(fake_A)
        cycle_loss_B = self.model.cycle_criterion(real_B, reconstructed_B)

        return (cycle_loss_A + cycle_loss_B) / 2

    def _calculate_losses(self, real_A, real_B, valid, lambda_id, lambda_cyc):
        identity_loss = self._calculate_identity_loss(real_A, real_B)
        loss_GAN, fake_A, fake_B = self._calculate_GAN_loss(real_A, real_B, valid)
        cycle_loss = self._calculate_cycle_loss(real_A, real_B, fake_A, fake_B)

        # computing the total loss
        total_loss = loss_GAN + lambda_id * identity_loss + lambda_cyc * cycle_loss
        total_loss.backward()
        self.model.optim_G.step()

        return total_loss, fake_A, fake_B

    def _train_disc_A(self, real_A, fake_A, valid, fake):
        self.model.optim_disc_A.zero_grad()

        real_loss = self.model.criterion_GAN(self.model.disc_A(real_A), valid)

        fake_A_ = self.model.fake_A_buffer.push_and_pop(fake_A)
        fake_loss = self.model.criterion_GAN(self.model.disc_A(fake_A_.detach()), fake)

        loss_disc_A = (real_loss + fake_loss) / 2

        loss_disc_A.backward()
        self.model.optim_disc_A.step()

        return loss_disc_A

    def _train_disc_B(self, real_B, fake_B, valid, fake):
        self.model.optim_disc_B.zero_grad()

        real_loss = self.model.criterion_GAN(self.model.disc_B(real_B), valid)

        fake_B_ = self.model.fake_B_buffer.push_and_pop(fake_B)
        fake_loss = self.model.criterion_GAN(self.model.disc_B(fake_B_.detach()), fake)

        loss_disc_B = (real_loss + fake_loss) / 2

        loss_disc_B.backward()
        self.model.optim_disc_B.step()

        return loss_disc_B

    def _train_discriminators(self, real_A, real_B, fake_A, fake_B, valid, fake):
        # now we train discriminator A
        loss_disc_A = self._train_disc_A(real_A, fake_A, valid, fake)

        # now we train discriminator B
        loss_disc_B = self._train_disc_B(real_B, fake_B, valid, fake)

        return (loss_disc_A + loss_disc_B) / 2

    def train_model(self, lambda_id, lambda_cyc, start_epoch=0, sample_interval=100):
        start = time.time()

        if start_epoch >= self.epochs:
            raise Exception("Starting Epoch exceeds Max Epochs")

        for epoch in range(start_epoch, self.epochs):
            print(f"Currently training on Epoch {epoch+1}")

            for i, batch in enumerate(self.training_data_loader):
                real_A = Variable(batch["A"].type(Tensor))
                real_B = Variable(batch["B"].type(Tensor))

                valid = Variable(
                    Tensor(np.ones((real_A.size(0), *self.model.disc_A.output_shape))),
                    requires_grad=True,
                )
                fake = Variable(
                    Tensor(np.zeros((real_A.size(0), *self.model.disc_A.output_shape))),
                    requires_grad=True,
                )
                
                print(real_A.shape, real_B.shape)

                # set gans to train mode
                self.model.gen_AB.train()  # gen_AB(real_A) = fake_B
                self.model.gen_BA.train()  # gen_BA(real_B) = fake_A

                # clean passed gradients values
                self.model.optim_G.zero_grad()

                # we start by computing the losses
                total_loss, fake_A, fake_B = self._calculate_losses(
                    real_A, real_B, valid, lambda_id, lambda_cyc
                )

                loss_D = self._train_discriminators(
                    real_A, real_B, fake_A, fake_B, valid, fake
                )

                batches_done = epoch * len(self.training_data_loader) + i
                batches_left = (
                    self.epochs * len(self.training_data_loader) - batches_done
                )

                time_left = datetime.timedelta(
                    seconds=batches_left * (time.time() - start)
                )
                start = time.time()

                print(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
                    % (
                        epoch,
                        self.epochs,
                        i,
                        len(self.training_data_loader),
                        loss_D.item(),
                        total_loss.item(),
                        time_left,
                    )
                )

                # If at sample interval save image
            if batches_done % sample_interval == 0:
                clear_output()
                self.plot_output(self.save_img_samples(batches_done), 30, 40)

    def plot_loss(self):
        step_bins = 20

        num_examples = (len(self.loss_from_generator_model) // step_bins) * step_bins
        plt.plot(
            range(num_examples // step_bins),
            torch.Tensor(self.loss_from_generator_model[:num_examples])
            .view(-1, step_bins)
            .mean(1),
            label="Generator Loss",
        )
        plt.plot(
            range(num_examples // step_bins),
            torch.Tensor(self.loss_from_discriminator_model[:num_examples])
            .view(-1, step_bins)
            .mean(1),
            label="Discriminator Loss",
        )
        plt.legend()
        plt.show()

        plt.savefig(f"{report_dir}/loss_plot.png")

    def plot_output(path, x, y):
        img = mpimg.imread(path)
        plt.figure(figsize=(x, y))
        plt.imshow(img)
        plt.show()

    def save_img_samples(self, batches_done):
        """Saves a generated sample from the test set"""
        print("batches_done ", batches_done)
        imgs = next(iter(self.val_dataloader))
        self.model.gen_AB.eval()
        self.model.gen_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = self.model.gen_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = self.model.gen_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=16, normalize=True)
        real_B = make_grid(real_B, nrow=16, normalize=True)
        fake_A = make_grid(fake_A, nrow=16, normalize=True)
        fake_B = make_grid(fake_B, nrow=16, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        path = report_dir + "/%s.png" % (
            batches_done
        )  # Path when running in Google Colab
        # path =  '/kaggle/working' + "/%s.png" % (batches_done)    # Path when running inside Kaggle
        save_image(image_grid, path, normalize=False)
        return path
