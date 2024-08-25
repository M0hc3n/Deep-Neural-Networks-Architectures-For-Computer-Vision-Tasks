import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from torchvision.utils import make_grid

from core.config import report_dir, device


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
        noise_shape,
        images_shape,
        criterion,
        epochs,
    ):
        self.training_data_loader = training_data_loader

        self.model = model

        self.criterion = criterion

        self.epochs = epochs

        self.noise_shape = noise_shape
        self.images_shape = images_shape

    def train_model(self):
        step = 0

        for epoch in range(self.epochs):
            print(f"Currently training on Epoch {epoch+1}")

            for real, labels in tqdm(self.training_data_loader):
                curr_batch_size = len(real)

                real = real.to(device)

                one_hot_labels = torch.nn.functional.one_hot(
                    labels.to(device), num_classes=self.model.num_classes
                )  # this should be (128, 10) (as labels are represented in one hot encoding)

                image_one_hot_labels = one_hot_labels[:, :, None, None]
                image_one_hot_labels = image_one_hot_labels.repeat(
                    1, 1, self.images_shape[1], self.images_shape[2]
                )

                # Let's start by training the Discriminator
                self.model.disc_opt.zero_grad()

                fake_noise = torch.randn(
                    curr_batch_size, self.noise_shape, device=device
                )
                noise_and_labels = torch.cat(
                    (fake_noise.float(), one_hot_labels.float()), 1
                )  # we concatentate fake noise (z) with y wrt to dimension 1 to form generator's input
                fake = self.model.gen(noise_and_labels)

                # now, we prepare the input for the discriminator
                # we need to attach y to both fake and real images
                # then, feed it to the discriminator
                fake_image_and_labels = torch.cat(
                    (fake.float(), image_one_hot_labels.float()), 1
                )
                real_image_and_labels = torch.cat(
                    (real.float(), image_one_hot_labels.float()), 1
                )

                # we generate the predictions:
                disc_fake_pred = self.model.disc(fake_image_and_labels.detach())  #!
                disc_real_pred = self.model.disc(real_image_and_labels)

                # we calculate the loss:
                disc_fake_loss = self.criterion(
                    disc_fake_pred, torch.ones_like(disc_fake_pred)
                )
                disc_real_loss = self.criterion(
                    disc_real_pred, torch.ones_like(disc_real_pred)
                )

                # we calculate the mean of losses:
                disc_loss = (disc_fake_loss + disc_real_loss) / 2

                # calculate the gradients and run a grad desc step
                disc_loss.backward(retain_graph=True)
                self.model.disc_opt.step()

                # store the losses:
                self.loss_from_discriminator_model += [disc_loss.item()]

                # we train the generator now:
                self.model.gen_opt.zero_grad()

                fake_image_and_labels = torch.cat(
                    (fake.float(), image_one_hot_labels.float()), 1
                )
                disc_fake_pred = self.model.disc(fake_image_and_labels)

                gen_loss = self.criterion(
                    disc_fake_pred, torch.ones_like(disc_fake_pred)
                )

                gen_loss.backward()
                self.model.gen_opt.step()

                self.loss_from_generator_model += [gen_loss.item()]

                if step % 50 == 0 and step > 0:
                    gen_mean = sum(self.loss_from_generator_model[-50:]) / 50

                    disc_mean = sum(self.loss_from_discriminator_model[-50:]) / 50

                    print(
                        f"Step {step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}"
                    )

                    self.plot_image(
                        image_tensor=real,
                        filename=f"{report_dir}/real_epoch_{epoch}_step_{step}.png",
                    )
                    self.plot_image(
                        image_tensor=fake,
                        filename=f"{report_dir}/fake_epoch_{epoch}_step_{step}.png",
                    )

                    self.plot_loss()
                step += 1

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

    def plot_image(
        self,
        image_tensor,
        num_images=25,
        nrow=5,
        show=False,
        filename=f"{report_dir}/example.png",
    ):
        image_tensor = (image_tensor + 1) / 2

        image_unflat = image_tensor.detach().cpu()

        image_grid = make_grid(image_unflat[:num_images], nrow=nrow)

        plt.imshow(image_grid.permute(1, 2, 0).squeeze())

        plt.savefig(filename)

        if show:
            plt.show()
