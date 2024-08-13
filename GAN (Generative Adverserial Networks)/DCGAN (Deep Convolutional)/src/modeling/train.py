import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch

from core.config import report_dir, device


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

    def train_model(self):
        for epoch in range(self.epochs):
            print(f"Currently training on Epoch {epoch+1}")

            for i, (real_images, _) in enumerate(self.training_data_loader):
                real_images = real_images.to(device)
                batch_size = real_images.size(0)

                # Generate fake images
                noise = torch.randn(batch_size, self.noise_shape, device=device)

                gen_images = self.model.generator(noise)

                # Create labels
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)

                # Train Discriminator
                self.model.optimizer_d.zero_grad()

                # Real images
                y_hat_on_real = self.model.discriminator(real_images)
                d_loss_real = self.criterion(y_hat_on_real, real_labels)
                d_loss_real.backward()

                # Fake images
                y_hat_on_fake = self.model.discriminator(gen_images.detach())
                d_loss_fake = self.criterion(y_hat_on_fake, fake_labels)
                d_loss_fake.backward()

                self.model.optimizer_d.step()

                # Train Generator
                self.model.optimizer_g.zero_grad()

                # this one is to compute loss to update the generator
                y_hat_for_generator = self.model.discriminator(gen_images)
                g_loss = self.criterion(y_hat_for_generator, real_labels)
                g_loss.backward()

                self.model.optimizer_g.step()

                # Append losses
                self.loss_from_discriminator_model.append(
                    d_loss_real.item() + d_loss_fake.item()
                )
                self.loss_from_generator_model.append(g_loss.item())

                if i % 100 == 0:
                    print(
                        f"\tCurrently training on batch number {i} of {len(self.training_data_loader)}"
                    )

            # Periodic sampling and visualization
            if epoch % 5 == 0:
                with torch.no_grad():
                    sample_noise = torch.randn(
                        self.num_sample, self.noise_shape, device=device
                    )
                    samples = self.model.generator(sample_noise).cpu().numpy()

                    plt.figure(figsize=(10, 4))
                    for k in range(self.num_sample):
                        plt.subplot(2, 5, k + 1)
                        plt.imshow(
                            (samples[k].transpose(1, 2, 0) + 1) / 2
                        )  # Normalize to [0, 1]
                        plt.xticks([])
                        plt.yticks([])

                    plt.savefig(f"{report_dir}/epoch-{epoch}.png")

            print(
                f"Epoch: {epoch+1}, Loss: D_real = {d_loss_real.item():.3f}, D_fake = {d_loss_fake.item():.3f}, G = {g_loss.item():.3f}"
            )

        print("Training completed with all epochs")

    # check those
    def plot_trainning_report(self):
        epochs = np.arange(self.epochs) + 1

        plt.figure(figsize=(20, 5))

        plt.subplot(121)

        plt.plot(epochs, self.train_losses, "blue", label="Training Losses")
        plt.plot(epochs, self.test_losses, "r", label="Test Losses")

        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title("Training and Test Losses vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.legend()
        plt.grid("off")
        plt.show()

        plt.subplot(122)

        plt.plot(epochs, self.train_accuracies, "blue", label="Training Accuracies")
        plt.plot(epochs, self.test_accuracies, "r", label="Test Accuracies")

        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

        plt.title("Training and Test Accuracies vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracies")
        plt.legend()
        plt.grid("off")
        plt.savefig(f"{report_dir}/classification_report.png")

    def generate_example(self):
        sample_noise = torch.randn(self.num_sample, self.noise_shape, device=device)
        samples = self.model.generator(sample_noise).cpu().numpy()

        plt.figure(figsize=(10, 4))
        for k in range(self.num_sample):
            plt.subplot(2, 5, k + 1)
            plt.imshow((samples[k].transpose(1, 2, 0) + 1) / 2)  # Normalize to [0, 1]
            plt.xticks([])
            plt.yticks([])

        plt.savefig(f"{report_dir}/example.png")
