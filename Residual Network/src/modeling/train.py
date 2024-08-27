import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

from modeling.metrics import accuracy, test_loss

import torch

from core.config import report_dir, device


class ModelTrainer:
    train_loss = []
    val_loss = []

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
        for epoch in range(self.epochs):
            curr_train_loss = 0
            correct_in_train = 0

            for inputs, labels in self.train_data:
                inputs, labels = inputs.to(device), labels.to(device)

                # set gradients to zero again, to not accumulate old gradients in the new clalculations
                self.optimizer.zero_grad()

                # generate predictions on the current set of inputs
                pred = self.model(inputs)

                # calculate loss
                loss = self.criterion(pred, labels)

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

            train_acc = correct_in_train / num_train_samples

            # Now check trained weights on the validation set
            val_running_loss = 0
            correct_val = 0

            self.model.eval().cuda()

            with torch.no_grad():
                for inputs, labels in self.validation_data:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass.
                    prediction = self.model(inputs)

                    # Compute the loss.
                    loss = self.criterion(prediction, labels)

                    # Compute validation accuracy.
                    _, predicted_outputs = torch.max(prediction.data, 1)
                    correct_val += (predicted_outputs == labels).float().sum().item()

                # Compute batch loss.
                val_running_loss += loss.data.item() * inputs.shape[0]

                val_epoch_loss = val_running_loss / val_samples_num
                self.val_loss.append(val_epoch_loss)
                val_acc = correct_val / val_samples_num

            info = "[Epoch {}/{}]: train-loss = {:0.6f} | train-acc = {:0.3f} | val-loss = {:0.6f} | val-acc = {:0.3f}"

            print(
                info.format(
                    epoch + 1,
                    self.epochs,
                    train_epoch_loss,
                    train_acc,
                    val_epoch_loss,
                    val_acc,
                )
            )

            torch.save(
                self.model.state_dict(), "/content/checkpoint_gpu_{}".format(epoch + 1)
            )

        torch.save(self.model.state_dict(), "/content/resnet-56_weights_gpu")

        return train_costs, val_costs
