import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from core.config import report_dir, device


class ModelTrainer:
    train_loss = []
    val_loss = []

    train_acc = []
    val_acc = []

    def __init__(
        self,
        model,
        train_data,
        validation_data,
        test_data,
        optimizer,
        criterion,
        epochs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

        self.epochs = epochs

    def fit(self):
        num_train_samples = len(self.train_data)
        num_validation_samples = len(self.validation_data)

        for epoch in range(self.epochs):
            curr_train_loss = 0
            correct_in_train = 0

            self.model.train()

            for inputs, labels in self.train_data:
                inputs, labels = inputs.to(device), labels.to(device)

                # set gradients to zero again, to not accumulate old gradients in the new clalculations
                self.optimizer.zero_grad()

                # generate predictions on the current set of inputs
                # recall that we also extract the predictions from the auxialiaries
                # and count them in the loss calculation
                pred, aux_pred1 = self.model(inputs)

                # calculate loss
                real_loss = self.criterion(pred, labels)
                loss_aux1 = self.criterion(aux_pred1, labels)

                loss = real_loss + (0.6 * loss_aux1)

                loss.backward()  # run backpropagation
                self.optimizer.step()  # compute the new weights

                # now store correctly predicted, and loss of this iteration

                # first, find the predicted class by looking for the maximum in the dimension 1
                _, predicted = torch.max(pred.data, 1)

                correct_in_train += (predicted == labels).float().sum().item()
                curr_train_loss += (
                    loss.data.item() * inputs.shape[0]
                )  # multiplied with batch length

            train_epoch_loss = curr_train_loss / num_train_samples
            self.train_loss.append(train_epoch_loss)

            train_acc_curr = correct_in_train / num_train_samples
            self.train_acc.append(train_acc_curr)

            # Now check trained weights on the validation set
            val_running_loss = 0
            correct_val = 0

            self.model.eval()

            with torch.no_grad():
                for inputs, labels in self.validation_data:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass.
                    prediction, prediction_aux1 = self.model(inputs)

                    # Compute the loss.
                    real_loss = self.criterion(prediction, labels)
                    loss_aux1 = self.criterion(prediction_aux1, labels)

                    loss = real_loss + (0.6 * loss_aux1)

                    # Compute validation accuracy.
                    _, predicted_outputs = torch.max(prediction.data, 1)
                    correct_val += (predicted_outputs == labels).float().sum().item()

                # Compute batch loss.
                val_running_loss += loss.data.item() * inputs.shape[0]

                val_epoch_loss = val_running_loss / num_validation_samples
                self.val_loss.append(val_epoch_loss)

                val_acc_curr = correct_val / num_validation_samples
                self.val_acc.append(val_acc_curr)

            info = "[Epoch {}/{}]: train-loss = {:0.6f} | train-acc = {:0.3f} | val-loss = {:0.6f} | val-acc = {:0.3f}"

            print(
                info.format(
                    epoch + 1,
                    self.epochs,
                    train_epoch_loss,
                    train_acc_curr,
                    val_epoch_loss,
                    val_acc_curr,
                )
            )

            torch.save(self.model.state_dict(), f"{report_dir}/checkpoint_{epoch + 1}")

        torch.save(self.model.state_dict(), f"{report_dir}/resnet-56_weights")

    def test(self):
        num_test_samples = len(self.test_data)
        correctly_classified = 0

        self.model.eval()

        # no gradient calculation on evaluation
        with torch.no_grad():
            for inputs, labels in self.test_data:
                inputs, labels = inputs.to(device), labels.to(device)

                pred = self.model(inputs)

                _, predicted_labels = torch.max(pred.data, 1)

                correctly_classified += (
                    (predicted_labels == labels).float().sum().item()
                )

        self.test_accuracy = correctly_classified / num_test_samples
        print("Test accuracy: {}".format(self.test_accuracy))

    def plot_trainning_report(self):
        epochs = np.arange(self.epochs) + 1

        plt.figure(figsize=(20, 5))

        plt.subplot(121)

        plt.plot(epochs, self.train_loss, "blue", label="Training Losses")
        plt.plot(epochs, self.val_loss, "r", label="Validation Losses")

        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title("Training and Validation Losses vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.legend()
        plt.grid("off")
        plt.show()

        plt.subplot(122)

        plt.plot(epochs, self.train_acc, "blue", label="Training Accuracies")
        plt.plot(epochs, self.val_acc, "r", label="Validation Accuracies")

        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

        plt.title("Training and Validation Accuracies vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracies")
        plt.legend()
        plt.grid("off")
        plt.savefig(f"{report_dir}/training_classification_report.png")

    def plot_testing_report(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.test_accuracy, "b", label="Testing Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Testing Accuracy Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.savefig(f"{report_dir}/testing_classification_report.png")
