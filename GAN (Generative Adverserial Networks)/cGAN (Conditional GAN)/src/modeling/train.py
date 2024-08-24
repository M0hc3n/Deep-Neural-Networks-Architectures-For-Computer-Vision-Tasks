import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch

from core.config import report_dir, device

import statistics


class ModelTrainer:
    training_data_loader = None

    model, criterion = None, None

    loss_from_discriminator_model = []
    loss_from_generator_model = []

    noise_and_labels = False
    fake = False

    fake_image_and_labels = False
    real_image_and_labels = False
    disc_fake_labels = False
    disc_real_labels = False

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

            for real, labels in enumerate(self.training_data_loader):
                curr_batch_size = len(real)

                real = real.to(device)

                torch.nn.functional.F.one_hot(
                    labels.to(device), num_classes=self.num_classes
                )
                
                # torch.cat((x.float(), y.float()), 1)

                pass

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

        plt.plot(
            epochs,
            statistics.mean(self.loss_from_discriminator_model),
            "blue",
            label="Loss From Discriminator (Mean of Batches)",
        )
        plt.plot(
            epochs,
            statistics.mean(self.loss_from_generator_model),
            "r",
            label="Loss From Generator (Mean of Batches)",
        )

        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title("Discriminator and Generator Losses vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.legend()
        plt.grid("off")
        plt.show()

        plt.subplot(122)

        plt.savefig(f"{report_dir}/classification_report.png")
