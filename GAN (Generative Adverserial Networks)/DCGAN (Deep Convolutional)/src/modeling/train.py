import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch

from core.config import report_dir, device

import statistics

class ModelTrainer:
    training_data_loader = None
    testing_data_loader = None

    model, criterion = None, None

    loss_from_discriminator_model = []
    loss_from_generator_model = []

    noise_shape = None

    epochs = 5

    def __init__(
        self,
        training_data_loader,
        model,
        num_sample,
        noise_shape,
        criterion,
        epochs,
    ):
        self.training_data_loader = training_data_loader

        self.model = model

        self.criterion = criterion

        self.epochs = epochs

        self.noise_shape = noise_shape
        self.num_sample = num_sample

    def train_model(self, generate_example):
        for epoch in range(self.epochs):
            print(f"Currently training on Epoch {epoch+1}")

            for i, (real_images, _) in enumerate(self.training_data_loader):
                # using .to() moves the current tensor to CPU or GPU memory
                # this operation is done for faster computation
                real_images = real_images.to(device)
                batch_size = real_images.size(0)

                # Our Training cycle starts by the following
                #
                # 1- generating fake images (from gaussian random noise, why ? cuz it WORKS !)
                noise = torch.randn(batch_size, self.noise_shape, device=device)

                gen_images = self.model.generator(noise)

                # 2- we create fake and real labels to be assigned afterwards for generator validation
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)

                # 3- we now start to train the discriminator:

                # We first need to invalidate the gradients calculated on the last batch (zero_grad() does the job)
                self.model.optimizer_d.zero_grad()

                # Now we call the discriminator on the real images, and compare the result with real_labels (which was generated in: 2)
                y_hat_on_real = self.model.discriminator(real_images)
                d_loss_real = self.criterion(y_hat_on_real, real_labels)
                # here, we calculate the gradients
                d_loss_real.backward()

                # Now we call the discriminator on the fake images this time, and compare the result with fake_labels (which was generated in: 2)
                y_hat_on_fake = self.model.discriminator(gen_images.detach())
                d_loss_fake = self.criterion(y_hat_on_fake, fake_labels)
                d_loss_fake.backward()

                # the last step in training the discriminator is to run the optimizer step (which would rely on gradient calculated on loss.backward()) to apply those gradients
                self.model.optimizer_d.step()

                # 4- we now start to train the generator:

                # we start by invalidating the last epoch computed gradients
                self.model.optimizer_g.zero_grad()

                # here, we compute loss to update the generator
                y_hat_for_generator = self.model.discriminator(gen_images)
                g_loss = self.criterion(y_hat_for_generator, real_labels)
                g_loss.backward()

                self.model.optimizer_g.step()

                # 5- we now save the losses (on both discr and gen) to plot them afterwards
                self.loss_from_discriminator_model.append(
                    d_loss_real.item() + d_loss_fake.item()
                )
                self.loss_from_generator_model.append(g_loss.item())

                print(
                    f"\tCurrently training on batch number {i + 1} of {len(self.training_data_loader)}"
                )

            # Periodic sampling and visualization
            if generate_example and epoch % 5 == 0:
                with torch.no_grad():
                    self.generate_example(filename=f"{report_dir}/epoch-{epoch}.png")

            print(
                f"Epoch: {epoch+1}, Loss: D_real = {d_loss_real.item():.3f}, D_fake = {d_loss_fake.item():.3f}, G = {g_loss.item():.3f}"
            )

        print("Training completed with all epochs")

    def generate_example(self, filename=f"{report_dir}/example.png"):
        sample_noise = torch.randn(self.num_sample, self.noise_shape, device=device)
        samples = self.model.generator(sample_noise).cpu().detach().numpy()

        plt.figure(figsize=(10, 4))
        for k in range(self.num_sample):
            plt.subplot(2, 5, k + 1)
            plt.imshow((samples[k].transpose(1, 2, 0) + 1) / 2)  # Normalize to [0, 1]
            plt.xticks([])
            plt.yticks([])

        plt.savefig(filename)

    def plot_trainning_report(self):
        epochs = np.arange(self.epochs) + 1

        plt.figure(figsize=(20, 5))

        plt.subplot(121)

        plt.plot(epochs, statistics.mean(self.loss_from_discriminator_model), "blue", label="Loss From Discriminator (Mean of Batches)")
        plt.plot(epochs, statistics.mean(self.loss_from_generator_model), "r", label="Loss From Generator (Mean of Batches)")

        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title("Discriminator and Generator Losses vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.legend()
        plt.grid("off")
        plt.show()

        plt.subplot(122)

        plt.savefig(f"{report_dir}/classification_report.png")
